from bayesopt.async_bo import AsyncBOTS
from .async_bo import AsyncBOHeuristicQEI, PLAyBOOK_LL, PLAyBOOK_H, PLAyBOOK_HL
from .async_bo import PLAyBOOK_L


class BatchBOHeuristicQEI(AsyncBOHeuristicQEI):
    pass


class BatchBOLocalPenalisation(PLAyBOOK_L):
    pass


class BatchBOLLP(PLAyBOOK_LL):
    pass


class BatchBOHLP(PLAyBOOK_H):
    pass


class BatchBOHLLP(PLAyBOOK_HL):
    pass


class BatchBOHeuristic(AsyncBOHeuristicQEI):
    pass


class BatchBOTS(AsyncBOTS):
    pass
