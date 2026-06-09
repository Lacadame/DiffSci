import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline

def interpolate_array(arr, method='linear'):
    """
    Interpolate an array [p0, p2, p4, p6] to [p0, p1, p2, p3, p4, p5, p6]
    by treating input as points at x-coordinates [0, 2, 4, 6] and interpolating
    to x-coordinates [0, 1, 2, 3, 4, 5, 6].
    
    Parameters:
    -----------
    arr : array-like
        Input array of values at even x-coordinates [p0, p2, p4, p6]
    method : str, optional
        Interpolation method: 'linear' or 'spline' (default: 'linear')
    
    Returns:
    --------
    numpy.ndarray
        Interpolated array [p0, p1, p2, p3, p4, p5, p6]
    """
    arr = np.asarray(arr)
    
    # Original x-coordinates (even indices: 0, 2, 4, 6, ...)
    n = len(arr)
    x_original = np.arange(0, n * 2, 2)
    
    # Target x-coordinates (all integers: 0, 1, 2, 3, 4, 5, 6, ...)
    x_target = np.arange(0, (n - 1) * 2 + 1)
    
    if method == 'linear':
        # Linear interpolation
        interp_func = interp1d(x_original, arr, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        result = interp_func(x_target)
    
    elif method == 'spline':
        # Cubic spline interpolation (s=0 means exact fit through points)
        spline = UnivariateSpline(x_original, arr, s=0, k=min(3, n-1))
        result = spline(x_target)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'spline'")
    
    return result


# Example usage
if __name__ == "__main__":
    # Example input: [p0, p2, p4, p6]
    input_array = np.array([1.0, 3.0, 5.0, 7.0])
    
    print("Input array:", input_array)
    print("Interpreted as points: (0, 1.0), (2, 3.0), (4, 5.0), (6, 7.0)")
    print()
    
    # Linear interpolation
    result_linear = interpolate_array(input_array, method='linear')
    print("Linear interpolation result:", result_linear)
    print("Values at x=[0,1,2,3,4,5,6]:", result_linear)
    print()
    
    # Spline interpolation
    result_spline = interpolate_array(input_array, method='spline')
    print("Spline interpolation result:", result_spline)
    print("Values at x=[0,1,2,3,4,5,6]:", result_spline)


