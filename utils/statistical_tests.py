"""
Statistical tests for forecasting significance and validation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr, normaltest, jarque_bera
import logging

logger = logging.getLogger(__name__)

class StatisticalTests:
    """
    Statistical tests for forecasting and data analysis
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical tests
        
        Args:
            significance_level: Alpha level for hypothesis tests
        """
        self.significance_level = significance_level
    
    def test_driver_significance(self, 
                                actuals: List[float], 
                                driver_forecast: List[float]) -> Dict[str, Any]:
        """
        Test if driver forecast is statistically significant for predicting actuals
        
        Args:
            actuals: Actual values
            driver_forecast: Driver forecast values
            
        Returns:
            Dictionary with significance test results
        """
        try:
            actuals_array = np.array(actuals)
            driver_array = np.array(driver_forecast)
            
            # Remove any NaN or infinite values
            mask = np.isfinite(actuals_array) & np.isfinite(driver_array)
            actuals_clean = actuals_array[mask]
            driver_clean = driver_array[mask]
            
            if len(actuals_clean) < 3:
                return {
                    'is_significant': False,
                    'reason': 'Insufficient valid data points for significance testing',
                    'p_value': 1.0,
                    'correlation': 0.0,
                    'test_type': 'insufficient_data'
                }
            
            # Pearson correlation test
            correlation, p_value = pearsonr(actuals_clean, driver_clean)
            
            # Additional tests
            spearman_corr, spearman_p = spearmanr(actuals_clean, driver_clean)
            
            # Linear regression significance test
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            X = driver_clean.reshape(-1, 1)
            y = actuals_clean
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # F-test for regression significance
            n = len(actuals_clean)
            f_statistic = (r2 / (1 - r2)) * (n - 2)
            f_p_value = 1 - stats.f.cdf(f_statistic, 1, n - 2)
            
            is_significant = (
                p_value < self.significance_level and 
                abs(correlation) > 0.1 and  # Minimum meaningful correlation
                f_p_value < self.significance_level
            )
            
            result = {
                'is_significant': is_significant,
                'p_value': float(p_value),
                'correlation': float(correlation),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'r_squared': float(r2),
                'f_statistic': float(f_statistic),
                'f_p_value': float(f_p_value),
                'significance_level': self.significance_level,
                'sample_size': int(n),
                'test_type': 'correlation_and_regression'
            }
            
            # Add interpretation
            if is_significant:
                if abs(correlation) > 0.7:
                    result['interpretation'] = 'Strong significant relationship'
                elif abs(correlation) > 0.5:
                    result['interpretation'] = 'Moderate significant relationship'
                else:
                    result['interpretation'] = 'Weak but significant relationship'
            else:
                result['interpretation'] = 'No significant relationship detected'
            
            return result
            
        except Exception as e:
            logger.error(f"Error in driver significance test: {str(e)}")
            return {
                'is_significant': False,
                'reason': f'Error in significance testing: {str(e)}',
                'p_value': 1.0,
                'correlation': 0.0,
                'test_type': 'error'
            }
    
    def test_forecast_residuals(self, 
                              actual: List[float], 
                              forecast: List[float]) -> Dict[str, Any]:
        """
        Test forecast residuals for normality and patterns
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            Dictionary with residual analysis results
        """
        try:
            actual_array = np.array(actual)
            forecast_array = np.array(forecast)
            
            # Calculate residuals
            residuals = actual_array - forecast_array
            
            # Remove any infinite values
            residuals_clean = residuals[np.isfinite(residuals)]
            
            if len(residuals_clean) < 3:
                return {
                    'normality_test': 'insufficient_data',
                    'autocorrelation_test': 'insufficient_data'
                }
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(residuals_clean)
            jb_stat, jb_p = jarque_bera(residuals_clean)
            
            # Autocorrelation test (Ljung-Box test approximation)
            # Simple lag-1 autocorrelation test
            if len(residuals_clean) > 1:
                lag1_residuals = residuals_clean[1:]
                lag0_residuals = residuals_clean[:-1]
                autocorr, autocorr_p = pearsonr(lag0_residuals, lag1_residuals)
            else:
                autocorr, autocorr_p = 0.0, 1.0
            
            # Residual statistics
            residual_stats = {
                'mean': float(np.mean(residuals_clean)),
                'std': float(np.std(residuals_clean)),
                'min': float(np.min(residuals_clean)),
                'max': float(np.max(residuals_clean)),
                'median': float(np.median(residuals_clean)),
                'skewness': float(stats.skew(residuals_clean)),
                'kurtosis': float(stats.kurtosis(residuals_clean))
            }
            
            return {
                'residual_statistics': residual_stats,
                'normality_test': {
                    'shapiro_statistic': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'is_normal_shapiro': shapiro_p > self.significance_level,
                    'jarque_bera_statistic': float(jb_stat),
                    'jarque_bera_p_value': float(jb_p),
                    'is_normal_jb': jb_p > self.significance_level
                },
                'autocorrelation_test': {
                    'lag1_correlation': float(autocorr),
                    'lag1_p_value': float(autocorr_p),
                    'has_autocorrelation': abs(autocorr) > 0.2 and autocorr_p < self.significance_level
                },
                'sample_size': len(residuals_clean)
            }
            
        except Exception as e:
            logger.error(f"Error in residual analysis: {str(e)}")
            return {
                'error': str(e),
                'normality_test': 'error',
                'autocorrelation_test': 'error'
            }
    
    def test_stationarity(self, data: List[float]) -> Dict[str, Any]:
        """
        Test time series for stationarity using Augmented Dickey-Fuller test
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            data_array = np.array(data)
            data_clean = data_array[np.isfinite(data_array)]
            
            if len(data_clean) < 4:
                return {
                    'is_stationary': False,
                    'reason': 'Insufficient data for stationarity test',
                    'test_type': 'insufficient_data'
                }
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(data_clean, autolag='AIC')
            
            adf_statistic = adf_result[0]
            adf_p_value = adf_result[1]
            critical_values = adf_result[4]
            
            is_stationary = adf_p_value < self.significance_level
            
            return {
                'is_stationary': is_stationary,
                'adf_statistic': float(adf_statistic),
                'adf_p_value': float(adf_p_value),
                'critical_values': {k: float(v) for k, v in critical_values.items()},
                'significance_level': self.significance_level,
                'test_type': 'augmented_dickey_fuller',
                'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
            }
            
        except ImportError:
            logger.warning("statsmodels not available, using simplified stationarity test")
            return self._simple_stationarity_test(data)
        except Exception as e:
            logger.error(f"Error in stationarity test: {str(e)}")
            return {
                'is_stationary': False,
                'reason': f'Error in stationarity test: {str(e)}',
                'test_type': 'error'
            }
    
    def _simple_stationarity_test(self, data: List[float]) -> Dict[str, Any]:
        """
        Simple stationarity test based on rolling statistics
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with simple stationarity test results
        """
        try:
            data_array = np.array(data)
            n = len(data_array)
            
            if n < 10:
                return {
                    'is_stationary': False,
                    'reason': 'Insufficient data for stationarity assessment',
                    'test_type': 'insufficient_data'
                }
            
            # Split data into two halves and compare means and variances
            mid = n // 2
            first_half = data_array[:mid]
            second_half = data_array[mid:]
            
            # T-test for mean difference
            t_stat, t_p = stats.ttest_ind(first_half, second_half)
            
            # F-test for variance difference
            f_stat = np.var(first_half) / np.var(second_half)
            f_p = 2 * min(stats.f.cdf(f_stat, len(first_half)-1, len(second_half)-1),
                         1 - stats.f.cdf(f_stat, len(first_half)-1, len(second_half)-1))
            
            # Consider stationary if means and variances are not significantly different
            is_stationary = t_p > self.significance_level and f_p > self.significance_level
            
            return {
                'is_stationary': is_stationary,
                'mean_test_p_value': float(t_p),
                'variance_test_p_value': float(f_p),
                'first_half_mean': float(np.mean(first_half)),
                'second_half_mean': float(np.mean(second_half)),
                'first_half_var': float(np.var(first_half)),
                'second_half_var': float(np.var(second_half)),
                'test_type': 'simple_split_test',
                'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
            }
            
        except Exception as e:
            logger.error(f"Error in simple stationarity test: {str(e)}")
            return {
                'is_stationary': False,
                'reason': f'Error in simple stationarity test: {str(e)}',
                'test_type': 'error'
            }
    
    def test_seasonality(self, data: List[float], period: int = 12) -> Dict[str, Any]:
        """
        Test for seasonal patterns in time series data
        
        Args:
            data: Time series data
            period: Expected seasonal period
            
        Returns:
            Dictionary with seasonality test results
        """
        try:
            data_array = np.array(data)
            n = len(data_array)
            
            if n < 2 * period:
                return {
                    'has_seasonality': False,
                    'reason': f'Insufficient data for seasonality test (need at least {2 * period} points)',
                    'test_type': 'insufficient_data'
                }
            
            # Create seasonal indices
            seasonal_means = []
            for i in range(period):
                seasonal_values = data_array[i::period]
                if len(seasonal_values) > 0:
                    seasonal_means.append(np.mean(seasonal_values))
                else:
                    seasonal_means.append(0)
            
            # Test if seasonal means are significantly different
            seasonal_groups = []
            for i in range(period):
                seasonal_values = data_array[i::period]
                if len(seasonal_values) > 0:
                    seasonal_groups.append(seasonal_values)
            
            if len(seasonal_groups) < 2:
                return {
                    'has_seasonality': False,
                    'reason': 'Insufficient seasonal groups for testing',
                    'test_type': 'insufficient_groups'
                }
            
            # One-way ANOVA test for seasonal differences
            try:
                f_stat, f_p = stats.f_oneway(*seasonal_groups)
                has_seasonality = f_p < self.significance_level
            except:
                # Fallback: compare variance of seasonal means to overall variance
                overall_mean = np.mean(data_array)
                seasonal_variance = np.var(seasonal_means)
                overall_variance = np.var(data_array)
                
                # Simple heuristic: if seasonal means vary significantly
                has_seasonality = seasonal_variance > (overall_variance * 0.1)
                f_stat, f_p = 0.0, 1.0
            
            return {
                'has_seasonality': has_seasonality,
                'f_statistic': float(f_stat),
                'f_p_value': float(f_p),
                'seasonal_means': [float(x) for x in seasonal_means],
                'seasonal_period': period,
                'test_type': 'anova_seasonal',
                'interpretation': 'Seasonal pattern detected' if has_seasonality else 'No clear seasonal pattern'
            }
            
        except Exception as e:
            logger.error(f"Error in seasonality test: {str(e)}")
            return {
                'has_seasonality': False,
                'reason': f'Error in seasonality test: {str(e)}',
                'test_type': 'error'
            }

