"""Test module for boolean operations."""
import numpy as np

import meshio
import pygmsh
from helpers import compute_volume


def geo_cube(geo_object):
    """Construct square using built in geometry."""
    cube = geo_object.add_box([-0.5, -0.5, -0.5], [1, 1, 1], 0.1)
    return geo_object, cube


def geo_builtin_cube(geo_object):
    """Construct a cube from built in geometries."""
    facets = []
    points = [[0, 0, 0],
              [1, 0, 0],
              [1, 1, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0, 1],
              [1, 1, 1],
              [0, 1, 1]]
    all_edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 7], [7, 6],
                 [6, 5], [5, 4], [0, 4], [5, 1], [2, 6], [7, 3]]
    all_facets = [[1, 2, 3, 4], [5, 6, 7, 8], [9, -8, 10, -1],
                  [11, -6, 12, -3], [-10, -7, -11, -2], [-12, -5, -9, -4]]

    pypoints = [geo_object.add_point(point, 0.1) for point in points]
    all_lines = [geo_object.add_line(pypoints[edge[0]], pypoints[edge[1]]) for edge in all_edges]
    for facet in all_facets:
        facet_lines = [-all_lines[abs(idxs) - 1] if idxs < 0 else
                       all_lines[abs(idxs) - 1] for idxs in facet]

        surface = geo_object.add_surface(geo_object.add_line_loop(facet_lines))
        facets.append(surface)
    cube = geo_object.add_volume(geo_object.add_surface_loop(facets))
    return geo_object, cube


def geo_sphere(geo_object):
    """construct sphere"""
    sphere = geo_object.add_ball([0, 0, 0], 0.1)
    return geo_object, sphere


def geos_diff():
    """Cconstruct surface using boolean fragments."""

    geo_object = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo_object, cube = geo_cube(geo_object)
    geo_object, sphere = geo_sphere(geo_object)

    geo_object.boolean_difference([cube], [sphere])

    geo_object2 = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo_object2, cube2 = geo_builtin_cube(geo_object2)
    geo_object2, sphere2 = geo_sphere(geo_object2)

    geo_object2.boolean_difference([cube2], [sphere2])

    return geo_object, geo_object2


def geos_frag():
    """Cconstruct surface using boolean fragments."""

    geo_object = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo_object, cube = geo_cube(geo_object)
    geo_object, sphere = geo_sphere(geo_object)

    geo_object.boolean_fragments([cube], [sphere])

    geo_object2 = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo_object2, cube2 = geo_builtin_cube(geo_object2)
    geo_object2, sphere2 = geo_sphere(geo_object2)

    geo_object2.boolean_fragments([cube2], [sphere2])
    return geo_object, geo_object2


def test_compare_cubes():
    """Test builtin and opencascade cubes."""
    geo1 = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo2 = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo1, _ = geo_cube(geo1)
    geo2, _ = geo_builtin_cube(geo2)
    for geo in [geo2, geo2]:
        mesh = pygmsh.generate_mesh(geo)
        assert np.abs(compute_volume(mesh) - 1) / 1 < 1e-3


def test_cube_sphere_hole():
    """Test planar surface with holes.

    Construct it with boolean operations and verify that it is the same.
    """
    for geo in geos_diff():
        mesh = pygmsh.generate_mesh(geo)
        surf = 1 - 0.1 ** 3 * np.pi * 4 / 3
        assert np.abs((compute_volume(mesh) - surf) / surf) < 1e-3
        return


def test_cube_sphere_slice():
    """Test planar suface square with circular hole.

    Also test for surface area of fragments.
    """
    for geo_object in geos_frag():

        # Gmsh 4 default format MSH4 doesn't have geometrical entities.
        mesh = pygmsh.generate_mesh(geo_object, extra_gmsh_arguments=["-format", "msh2"])
        assert np.abs((compute_volume(mesh) - 1 / 1)) < 1e-3
        ref = 1
        val = compute_volume(mesh)
        assert np.abs(val - ref) < 1e-3 * ref
        outer_mask = np.where(mesh.cell_data["tetra"]["gmsh:geometrical"] == 14)[0]
        outer_cells = {}
        outer_cells["tetra"] = mesh.cells["tetra"][outer_mask]

        inner_mask = np.where(mesh.cell_data["tetra"]["gmsh:geometrical"] == 13)[0]
        inner_cells = {}
        inner_cells["tetra"] = mesh.cells["tetra"][inner_mask]

        ref = 1 - 0.1 ** 3 * np.pi * 4 / 3
        value = compute_volume(meshio.Mesh(mesh.points, outer_cells))
        assert np.abs(value - ref) < 1e-2 * ref
        return


def test_diff_union():
    """Test planar surface with holes.

    Construct it with boolean operations and verify that it is the same.
    """
    # construct surface using boolean
    geo_object = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo_object, vol1 = geo_cube(geo_object)
    geo_object, vol2 = geo_sphere(geo_object)

    geo_object.add_physical([vol1], label=1)
    geo_object.add_physical([vol2], label=2)
    diff = geo_object.boolean_difference([vol1], [vol2], delete_other=False)
    geo_object.boolean_union([diff, vol2])
    mesh = pygmsh.generate_mesh(geo_object)
    assert np.abs((compute_volume(mesh) - 1) / 1) < 1e-3
    surf = 1 - 0.1 ** 3 * np.pi * 4 / 3
    outer_mask = np.where(mesh.cell_data["tetra"]["gmsh:physical"] == 1)[0]
    outer_cells = {}
    outer_cells["tetra"] = mesh.cells["tetra"][outer_mask]

    inner_mask = np.where(mesh.cell_data["tetra"]["gmsh:physical"] == 2)[0]
    inner_cells = {}
    inner_cells["tetra"] = mesh.cells["tetra"][inner_mask]

    value = compute_volume(meshio.Mesh(mesh.points, outer_cells))
    assert np.abs((value - surf)) < 1e-2 * surf
    return


def test_diff_physical_assignment():
    """ construct surface using boolean.

    Ensure that after a difference operation the initial volume physical label
    is kept for the operated geometry.
    """
    geo_object = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo_object, vol1 = geo_cube(geo_object)
    geo_object, vol2 = geo_sphere(geo_object)

    geo_object.add_physical([vol1], label=1)
    diff = geo_object.boolean_difference([vol1], [vol2])

    mesh = pygmsh.generate_mesh(geo_object)
    assert np.allclose(
        mesh.cell_data["tetra"]["gmsh:physical"],
        np.ones(mesh.cells["tetra"].shape[0]),
    )
    surf = 1 - 0.1 ** 3 * np.pi * 4 / 3
    assert np.abs((compute_volume(mesh) - surf) / surf) < 1e-3
    return


def test_diff_physical_assignment_with_builtin():
    """ construct surface using boolean.

    Ensure that after a difference operation the initial volume physical label
    is kept for the operated geometry.
    """
    geo_object = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo_object, vol1 = geo_builtin_cube(geo_object)
    geo_object, vol2 = geo_sphere(geo_object)

    geo_object.add_physical([vol1], label=1)
    diff = geo_object.boolean_difference([vol1], [vol2])

    mesh = pygmsh.generate_mesh(geo_object)
    assert np.allclose(
        mesh.cell_data["tetra"]["gmsh:physical"],
        np.ones(mesh.cells["tetra"].shape[0]),
    )
    surf = 1 - 0.1 ** 3 * np.pi * 4 / 3
    assert np.abs((compute_volume(mesh) - surf) / surf) < 1e-2
    return


def test_diff_union_with_builtin():
    """Test planar surface with holes.

    Construct it with boolean operations and verify that it is the same.
    """
    # construct surface using boolean
    geo_object = pygmsh.opencascade.Geometry(0.1, 0.1)
    geo_object, vol1 = geo_builtin_cube(geo_object)
    geo_object, vol2 = geo_sphere(geo_object)

    geo_object.add_physical([vol1], label=1)
    geo_object.add_physical([vol2], label=2)
    diff = geo_object.boolean_difference([vol1], [vol2], delete_other=False)
    geo_object.boolean_union([diff, vol2])
    mesh = pygmsh.generate_mesh(geo_object)
    assert np.abs((compute_volume(mesh) - 1) / 1) < 1e-2
    surf = 1 - 0.1 ** 3 * np.pi * 4 / 3
    print(mesh.cell_data)
    outer_mask = np.where(mesh.cell_data["tetra"]["gmsh:physical"] == 1)[0]
    outer_cells = {}
    outer_cells["tetra"] = mesh.cells["tetra"][outer_mask]

    inner_mask = np.where(mesh.cell_data["tetra"]["gmsh:physical"] == 2)[0]
    inner_cells = {}
    inner_cells["tetra"] = mesh.cells["tetra"][inner_mask]

    value = compute_volume(meshio.Mesh(mesh.points, outer_cells))
    assert np.abs((value - surf)) < 1e-2 * surf
    return


def test_slice_diff():
    """Test planar surface with holes.

    Construct it with boolean operations and verify that it is the same.
    """
    # construct surface using boolean
    geo_object = pygmsh.opencascade.Geometry(0.1, 0.1)
    geo_object, vol1 = geo_cube(geo_object)
    geo_object, vol2 = geo_sphere(geo_object)
    geo_object, vol3 = geo_sphere(geo_object)

    geo_object.add_physical([vol1], label=1)
    geo_object.add_physical([vol2], label=2)
    geo_object.add_physical([vol3], label=3)
    slce = geo_object.boolean_fragments([vol1], [vol2])
    diff = geo_object.boolean_difference([slce], [vol3])
    mesh = pygmsh.generate_mesh(geo_object)
    surf = 1 - 0.1 ** 3 * np.pi * 4 / 3
    assert np.abs((compute_volume(mesh) - surf) / surf) < 1e-2
    print(mesh.cell_data)
    outer_mask = np.where(mesh.cell_data["tetra"]["gmsh:physical"] == 1)[0]
    outer_cells = {}
    outer_cells["tetra"] = mesh.cells["tetra"][outer_mask]

    value = compute_volume(meshio.Mesh(mesh.points, outer_cells))
    assert np.abs((value - surf)) < 1e-2 * surf
    return


def test_fragments():
    """Test planar surface with holes.

    Construct it with boolean operations and verify that it is the same.
    """
    # construct surface using boolean
    geo_object = pygmsh.opencascade.Geometry(0.04, 0.04)
    geo_object, vol1 = geo_cube(geo_object)
    geo_object, vol2 = geo_sphere(geo_object)

    geo_object.add_physical([vol1], label=1)
    geo_object.add_physical([vol2], label=2)
    print(vol1.id)
    print(vol2.id)
    diff = geo_object.boolean_fragments([vol1], [vol2])
    print(diff.id)
    '''
    vol1.id = diff.id
    resulting_names = [diff.id]
    geo_object._GMSH_CODE.append(
        'Physical {}({}) = {{{}}};'.format(
            'Volume', 0, ', '.join([name for name in resulting_names])
        ))
    '''
    mesh = pygmsh.generate_mesh(geo_object)
    print(compute_volume(mesh))
    print(diff.is_list)
    print(diff.id)
    assert np.abs((compute_volume(mesh) - 1) / 1) < 1e-3
    surf = 1 - 0.1 ** 3 * np.pi * 4 / 3
    outer_mask = np.where(mesh.cell_data["tetra"]["gmsh:physical"] == 1)[0]
    outer_cells = {}
    outer_cells["tetra"] = mesh.cells["tetra"][outer_mask]

    inner_mask = np.where(mesh.cell_data["tetra"]["gmsh:physical"] == 2)[0]
    inner_cells = {}
    inner_cells["tetra"] = mesh.cells["tetra"][inner_mask]

    value = compute_volume(meshio.Mesh(mesh.points, outer_cells))
    assert np.abs((value - surf)) < 1e-2 * surf
    return


if __name__ == "__main__":
    #test_compare_cubes()
    #test_cube_sphere_hole()
    #test_cube_sphere_slice()
    #test_diff_union()
    #test_diff_physical_assignment_with_builtin()
    #test_diff_physical_assignment()

    #test_diff_union_with_builtin()
    #test_slice_diff()
    test_fragments()
