import numpy as np
from scipy import stats

def AB_test(test: np.ndarray, control: np.ndarray, confidence: float=0.95, h0=0, report:bool = False):
    """Uses different methods to calculate the information gained on series X from knowing the conditioning
    series Y

    Args:
        X: tokenized input 1D array.
        Y: full set of potential tokens that Y can take

    Returns:
        The information gain and result of statistical test
    """

    mu1, mu2 = np.mean(test), np.mean(control) #test.mean(), control.mean()
    se1, se2 = np.std(test) / np.sqrt(len(test)), np.std(control) / np.sqrt(len(control))
    
    diff = mu1 - mu2
    se_diff = np.sqrt(np.var(test)/len(test) + np.var(control)/len(control))
    
    # Probability of the difference in entropies being greater than h0
    z_stats = (diff-h0)/se_diff
    p_value = stats.norm.cdf(z_stats)
    
    def critial(se): return -se*stats.norm.ppf((1 - confidence)/2)
    
    print(f"Test {confidence*100}% CI: {mu1} +- {critial(se1)}")
    print(f"Control {confidence*100}% CI: {mu2} +- {critial(se2)}")
    print(f"Test-Control {confidence*100}% CI: {diff} +- {critial(se_diff)}")
    print(f"Z Statistic {z_stats}")
    print(f"P-Value {p_value}")

    return p_value > confidence

def classic_information_gain(X: np.ndarray, A: np.ndarray) -> tuple(float, bool):





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


