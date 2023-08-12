from ._derivatives import dReLU, dSigmoid, dTanh, dLeakyReLU

derivative_map = {
    "drelu": dReLU,
    "dsigmoid": dSigmoid,
    "dtanh": dTanh,
    "dleakyrelu": dLeakyReLU,
}