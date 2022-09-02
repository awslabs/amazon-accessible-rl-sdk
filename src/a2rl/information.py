import numpy as np
from scipy import stats


def AB_test(test: np.ndarray, control: np.ndarray, confidence: float=0.95, h0=0):
    """Uses T test to determine the difference between 2 distributions with certain confidence

    Args:
        test: tokenized input 1D array.
        control: full set of potential tokens that Y can take

    Returns:
        True if the the test and control samples are statistically significant
        False otherwise
    """

    mu1, mu2 = np.mean(test), np.mean(control) 

    diff = mu1 - mu2
    se_diff = np.sqrt(np.var(test)/len(test) + np.var(control)/len(control))
    
    z_stats = (diff-h0)/se_diff
    p_value = stats.norm.cdf(z_stats)

    return p_value > confidence


def entropy(Y: np.ndarray, token_space: np.ndarray = None) -> float:
    """The `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_ of the input
    series. Normalizes for the number of coditioning tokens

    Args:
        Y: tokenized input 1D array.
        token_space: full set of potential tokens that Y can take

    Returns:
        The normalized entropy or efficiency of Y
    """

    if token_space is None:
        n_tokens = len(Y)
        if n_tokens <= 1:
            return 0

        value,counts = np.unique(Y.astype("<U22"), return_counts=True)
        probs = counts / n_tokens

        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0

        ent = 0.
        for i in probs:
            ent -= i * np.log2(i)
        
        return ent / np.log2(n_classes)

    else:
        n_tokens = len(token_space)
        if n_tokens <= 1:
            return 0

        value,counts = np.unique(Y.astype("<U22"), return_counts=True)
        probs = counts / n_tokens
        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0

        ent = 0.
        for i in probs:
            ent -= i * np.log2(i)
        return ent / np.log2(n_classes)



def classic_information_gain(X: np.ndarray, A: np.ndarray, normalization: str, baseline: float) -> tuple(float, bool):
    """ Uses Shannons entropy formula to calculate the information gain.
    
    Args:
        X: tokenized input 1D array.
        A: Conditioning 1D groupby for X

    Returns:
        (information gain, test passed)
    """

    if normalization == '':




def conditional_information_test(X: np.ndarray, A: np.ndarray, method: str) -> tuple(float, bool):
    """Uses different methods to calculate the information gained on series X from knowing the conditioning
    series Y

    Args:
        X: tokenized input 1D array.
        Y: full set of potential tokens that Y can take

    Returns:
        The information gain and result of statistical test
    """


    return information_gain, test_passed


