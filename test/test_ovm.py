import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.quad_tri_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        #helpers.add_cell_data(helpers.tri_mesh, [("medit:ref", (), int)]),
    ],
)
def test_io(mesh):
    helpers.write_read(meshio.ovm.write, meshio.ovm.read, mesh, 1.0e-15, extension=".ovm")

