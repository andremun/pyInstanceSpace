import numpy as np
from numpy.typing import NDArray
import stage


class siftedStage(stage):
    def __init__(self, 
                 
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SiftedOptions
        ) -> None:
        
        self.x = x
        self.y = y
        self.y_bin = y_bin
        self.opts = opts
        
    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        return [
            ["x", NDArray[np.double]],
            ["y", NDArray[np.double]],
            ["y_bin", NDArray[np.bool_]],
            ["opts", SiftedOptions]
        ]
    
    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        return [
            ["flag", int],
            ["rho", np.double],
            ["k", int],
            ["n_trees", int],
            ["max_lter", int],
            ["replicates", int],
            ["idx", NDArray[np.int_]]
        ]
    
    @staticmethod
    def _run(self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SiftedOptions
        ) -> tuple[int, np.double, int, int, int, int, NDArray[np.int_]]:

        raise NotImplementedError
    
    def sifted(self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SiftedOptions
        ) -> tuple[int, np.double, int, int, int, int, NDArray[np.int_]]:
        
        raise NotImplementedError