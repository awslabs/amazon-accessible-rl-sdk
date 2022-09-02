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
    series. 
    
    Credit:@Dashora7
    token_space: boolean. This determines whether to use the probability of correct prediction rather than entropy in the splits. 
    This essentially alters the calculation by normalizing by the total number of elements.

    Args:
        Y: tokenized input 1D array.
        token_space: set of splitting tokens of Y

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


def classic_information_gain(X: np.ndarray, A: np.ndarray, baseline: float):
    """ Uses Shannons entropy formula to calculate the information gain.
    
    Args:
        X: tokenized input 1D array.
        A: Conditioning 1D groupby for X

    Returns:
        (information gain, test passed)
        test passed is True if the information gain is above the baseline.
    """

    z = np.vstack((A, X)).T
    z = z[z[:, 0].argsort()]
    groups = np.split(z[:, 1], np.unique(z[:, 0], return_index=True)[1][1:])

    values, counts = np.unique(z[:, 0], return_counts=True)
    #entropies = np.array([entropy(g, token_space=X) for g in groups])

    entropies = np.array([entropy(g) for g in groups])
        
    probs = counts / np.sum(counts)

    _infomation_gained = entropy(X) - np.sum(probs * entropies)
    _test_passed = _infomation_gained > baseline

    return (_infomation_gained, _test_passed)

def normalised_information_gain(X: np.ndarray, A: np.ndarray, baseline: float):
    """ Uses Shannons entropy formula to calculate the information gain but normalizes it by the len of A
    
    Args:
        X: tokenized input 1D array.
        A: Conditioning 1D groupby for X

    Returns:
        (information gain, test passed)
        test passed is True if the information gain is above the baseline.
    """

    z = np.vstack((A, X)).T
    z = z[z[:, 0].argsort()]
    groups = np.split(z[:, 1], np.unique(z[:, 0], return_index=True)[1][1:])

    values, counts = np.unique(z[:, 0], return_counts=True)
    entropies = np.array([entropy(g, token_space=A) for g in groups])
        
    probs = counts / np.sum(counts)

    _infomation_gained = entropy(X) - np.sum(probs * entropies)
    _test_passed = _infomation_gained > baseline

    return _infomation_gained, _test_passed

def group_entropies(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """ Return an array of the entropies of the group split up by A
    
    Args:
        X: tokenized input 1D array.
        A: Conditioning 1D groupby for X

    Returns:
        The array of entropies of the group
    """

    z = np.vstack((A, X)).T
    z = z[z[:, 0].argsort()]
    groups = np.split(z[:, 1], np.unique(z[:, 0], return_index=True)[1][1:])
    
    entropies = np.array([entropy(g, token_space=A) for g in groups])
    
    return entropies

def placebo_action(X: np.ndarray, A: np.ndarray):
    """ Tests if there is a statistical difference between the entropies of the original H(X|A)
    versus H(X|shuffled(A)) which destroys the structure.

    We just shuffle A to keep the same number of groups and then run a T-test
    
    Args:
        X: tokenized input 1D array.
        A: Conditioning 1D groupby for X

    Returns:
        (information gain, test passed)
        test passed is True if the information gain if the T-test works
    """
    # We permute the conditioning variable 

    original = group_entropies(X, A)
    destroyed_structure = group_entropies(X,np.random.permutation(A))

    _infomation_gained = np.mean(destroyed_structure) - np.mean(original)
    _test_passed = AB_test(original,destroyed_structure)

    return _infomation_gained, _test_passed


def conditional_information_test(X: np.ndarray, A: np.ndarray, method: str):
    """Uses different methods to calculate the information gained on series X from knowing the conditioning
    series Y

    Args:
        X: tokenized input 1D array.
        Y: full set of potential tokens that Y can take

    Returns:
        The information gain and result of statistical test
    """
    if method == 'norm':

        return normalised_information_gain(X,A, 0.5)

    elif method == 'placebo':
        
        return placebo_action(X,A)

    else:
        return classic_information_gain(X,A, 0.5)


