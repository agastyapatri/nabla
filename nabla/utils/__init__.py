from ._derivatives import dReLU, dSigmoid, dTanh, dLeakyReLU
from .structures import Structure
from .activations import * 


derivative_map = {
    "drelu": dReLU,
    "dsigmoid": dSigmoid,
    "dtanh": dTanh,
    "dleakyrelu": dLeakyReLU,
}
