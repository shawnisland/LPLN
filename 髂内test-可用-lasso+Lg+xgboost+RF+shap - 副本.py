# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (roc_curve, auc, accuracy_score, f1_score, 
                           cohen_kappa_score, recall_score, precision_score)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, 
                             recall_score, precision_score)


# %%
class ClinicalPredictionPipeline:
    def __init__(self, random_state=42):
        self.output_dir = 'output_figures'  
        os.makedirs(self.output_dir, exist_ok=True)  
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_selector = LassoCV(cv=5, random_state=random_state)
        self.models = {

            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state),
            'XGBoost': xgb.XGBClassifier(random_state=random_state),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,  
                weights='uniform',
                metric='minkowski',
                p=2  
            ),
            'SVM': SVC(
                kernel='rbf',  
                probability=True,  
                random_state=random_state,
                class_weight='balanced'  
            )
        }
        self.selected_features = None
        self.best_model = None
        self.best_model_name = None
        self.best_auc = 0



    # 
    def get_display_name(self, feature):
        
        return self.feature_names_map.get(feature, feature)

    def optimize_knn_parameters(self, X_train, y_train, X_val, y_val):
        
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'manhattan']
        }
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['KNN'] = KNeighborsClassifier(**grid_search.best_params_)
        

        
    def optimize_svm_parameters(self, X_train, y_train, X_val, y_val):
        
        from sklearn.model_selection import GridSearchCV
        

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
        
        
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=self.random_state),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
    
        grid_search.fit(X_train, y_train)
        
        
        best_params = grid_search.best_params_
        best_params['probability'] = True  # 
        best_params['random_state'] = self.random_state
        self.models['SVM'] = SVC(**best_params)
        
        print("SVM:", grid_search.best_params_)
        print( grid_search.best_score_)
    
    def plot_lasso_cv_curve(self):
        plt.figure(figsize=(10, 6))
        
        mse_path = self.feature_selector.mse_path_
        alphas = self.feature_selector.alphas_
        
        mean_mse = mse_path.mean(axis=1)
        std_mse = mse_path.std(axis=1)
        
        plt.plot(np.log10(alphas), mean_mse, label='Mean Squared Error', color='blue', lw=2)
        plt.fill_between(np.log10(alphas), mean_mse - std_mse, mean_mse + std_mse, alpha=0.2, color='blue')
        
        best_alpha = self.feature_selector.alpha_
        plt.axvline(np.log10(best_alpha), color='red', linestyle='--', label=f'Best Alpha: {best_alpha:.4f}')
        
        plt.xlabel('log10(alpha)')
        plt.ylabel('Mean Squared Error')
        plt.title('LASSO Cross-Validation Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/lasso_cv_curve.tiff', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_lasso_coefficient_path(self, X_scaled, y):
        alphas = np.logspace(-4, 2, 100)  # 
        
        coef_path = []
        
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_scaled, y)
            coef_path.append(lasso.coef_)
         
        coef_path = np.array(coef_path)
        
        plt.figure(figsize=(10, 6))
        for i in range(coef_path.shape[1]):
            plt.plot(np.log10(alphas), coef_path[:, i], label=f'Feture {i + 1}')
        
        best_alpha = self.feature_selector.alpha_
        plt.axvline(np.log10(best_alpha), color='red', linestyle='--', label=f'Best Alpha: {best_alpha:.4f}')
        
        plt.xlabel('log10(alpha)')
        plt.ylabel('Coefficients')
        plt.title('LASSO Coefficients Path')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/lasso_coefficient_path.tiff', dpi=300, bbox_inches='tight')
        plt.close()

    def feature_selection(self, X, y, threshold=0.01):
        
        X_scaled = self.scaler.fit_transform(X)
        self.feature_selector.fit(X_scaled, y)
        
        self.plot_lasso_cv_curve()
        
        self.plot_lasso_coefficient_path(X_scaled, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Display_Name': [self.get_display_name(f) for f in X.columns],
            'Coefficient': np.abs(self.feature_selector.coef_)
        }).sort_values('Coefficient', ascending=False)
        
        selected_mask = feature_importance['Coefficient'] > threshold
        self.selected_features = feature_importance[selected_mask]['Feature'].tolist()
        removed_features = feature_importance[~selected_mask]['Feature'].tolist()
        
        print(feature_importance.to_string(index=False))
        
        print(f"\n{threshold}")
        print(f"\nselected_features ({len(self.selected_features)}):")
        for feat in self.selected_features:
            coef = feature_importance[feature_importance['Feature'] == feat]['Coefficient'].values[0]
            display_name = feature_importance[feature_importance['Feature'] == feat]['Display_Name'].values[0]
            print(f"- {display_name}: {coef:.6f}")
        
        print(f"\nremoved_features ({len(removed_features)}):")
        for feat in removed_features:
            coef = feature_importance[feature_importance['Feature'] == feat]['Coefficient'].values[0]
            display_name = feature_importance[feature_importance['Feature'] == feat]['Display_Name'].values[0]
            print(f"- {display_name}: {coef:.6f}")
        
        plt.figure(figsize=(12, 8))
        display_importance = feature_importance.copy()
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
        plt.title('The results of Lasso feature selection')
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.tiff', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.selected_features

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):

        def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, alpha=0.05):

            scores = []
            for _ in range(n_bootstrap):
                # 
                y_true_resampled, y_pred_resampled = resample(y_true, y_pred, random_state=42)
                # 
                score = metric_func(y_true_resampled, y_pred_resampled)
                scores.append(score)
            
            lower = np.percentile(scores, 100 * alpha / 2)
            upper = np.percentile(scores, 100 * (1 - alpha / 2))
            metric_value = metric_func(y_true, y_pred)
            
            return metric_value, (lower, upper)


        metrics = {
            'accuracy_score': bootstrap_metric(y_true, y_pred, accuracy_score),
            'F1': bootstrap_metric(y_true, y_pred, f1_score),
            'Sensitivity': bootstrap_metric(y_true, y_pred, recall_score),
            'Specificity': bootstrap_metric(y_true, (y_pred_proba >= 0.5).astype(int), 
                                    lambda y_true, y_pred: (y_true == y_pred)[y_true == 0].mean()),
            'Positive': bootstrap_metric(y_true, y_pred, precision_score),
            'Negitive': bootstrap_metric(y_true, (y_pred_proba >= 0.5).astype(int), 
                                        lambda y_true, y_pred: (y_true == y_pred)[y_true == 0].mean())
        }
        return metrics

    def plot_roc_curve(self, y_true, y_pred_proba, title):
        """ROC"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'Receiver Operating Characteristic Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic Curve - {title}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(f'{self.output_dir}/roc_curve_{safe_title}.tiff', dpi=300, bbox_inches='tight')
        plt.close()
        
        return roc_auc

    def plot_calibration_curve(self, y_true, y_pred_proba, title):
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(10, 8))
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Predictive probability')
        plt.ylabel('Actual probability')
        plt.title(f'Calibration curve - {title}')
        plt.tight_layout()
        
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(f'{self.output_dir}/calibration_curve_{safe_title}.tiff', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_decision_curve(self, y_true, y_pred_proba, title):
        thresholds = np.arange(0.01, 1, 0.01)
        net_benefits = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n = len(y_true)
            
            net_benefit = (tp/n) - (fp/n) * (threshold/(1-threshold))
            net_benefits.append(net_benefit)
        
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, net_benefits, label='Model')
        plt.xlabel('Threshold probability')
        plt.ylabel('Net Benefit')
        plt.title(f'Decision Curve Analysis - {title}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(f'{self.output_dir}/decision_curve_{safe_title}.tiff', dpi=300, bbox_inches='tight')
        plt.close()

    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        X_train = X_train[self.selected_features]
        X_val = X_val[self.selected_features]
        X_test = X_test[self.selected_features]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.optimize_knn_parameters(X_train_scaled, y_train, X_val_scaled, y_val)
        self.optimize_svm_parameters(X_train_scaled, y_train, X_val_scaled, y_val)
        

        results = {}
        predictions = {}
        
        for name, model in self.models.items():
            print(f"\ntrain {name}...")
            
            model.fit(X_train_scaled, y_train)
            
            sets = {
                'Training': (X_train_scaled, y_train),
                'Validation': (X_val_scaled, y_val),
                'Test': (X_test_scaled, y_test)
            }
            
            model_results = {}
            model_predictions = {}
            
            for set_name, (X, y) in sets.items():
                y_pred_proba = model.predict_proba(X)[:, 1]
                y_pred = model.predict(X)
                
                model_predictions[set_name] = pd.DataFrame({
                    'true_label': y,
                    'predicted_prob': y_pred_proba
                })
                
                metrics = self.calculate_metrics(y, y_pred, y_pred_proba)
                model_results[set_name] = metrics
                
                print(f"\n{name} - {set_name}:")
                for metric_name, (value, (lower, upper)) in metrics.items():
                    print(f"{metric_name}: {value:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
                
                auc = self.plot_roc_curve(y, y_pred_proba, f"{name} - {set_name}")
                self.plot_calibration_curve(y, y_pred_proba, f"{name} - {set_name}")
                self.plot_decision_curve(y, y_pred_proba, f"{name} - {set_name}")
                
                if set_name == 'Test' and auc > self.best_auc:
                    self.best_auc = auc
                    self.best_model = model
                    self.best_model_name = name
            
            results[name] = model_results
            predictions[name] = model_predictions

        for name, pred_dict in predictions.items():
            with pd.ExcelWriter(f'{name}_predictions.xlsx') as writer:
                for set_name, df in pred_dict.items():
                    df.to_excel(writer, sheet_name=set_name, index=False)
        
        metrics_data = []
        for name, result_dict in results.items():
            for set_name, metrics in result_dict.items():
                for metric_name, (value, (lower, upper)) in metrics.items():
                    metrics_data.append({
                        'model': name,
                        'set': set_name,
                        'metric_name': metric_name,
                        'value': value,
                        'Lower 95%': lower,
                        'Upper 95%': upper
                    })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel('evaluation_metrics_with_ci.xlsx', index=False)
        
        return results, predictions

    def explain_best_model(self, X_test):
        print(f"\nSHAP {self.best_model_name}...")
        X_test = X_test[self.selected_features]
        X_test_scaled = self.scaler.transform(X_test)
        
        
        X_test_scaled_df = pd.DataFrame(X_test_scaled, 
                                    columns=self.selected_features)
          
        display_names = [self.get_display_name(f) for f in self.selected_features]
        
        if hasattr(self, 'feature_name_map') and self.feature_name_map:
            
            display_columns = []
            for col in self.selected_features:
                if col in self.feature_name_map:
                    display_columns.append(self.feature_name_map[col])
                else:
                    display_columns.append(col)
            
            X_test_display = pd.DataFrame(X_test_scaled, columns=display_columns)
        else:
            X_test_display = X_test_scaled_df
        
        try:
            explainer = shap.TreeExplainer(self.best_model)
            shap_explanations = explainer(X_test_scaled_df)
            if isinstance(shap_explanations, shap.Explanation):
                shap_values = shap_explanations.values
                if len(shap_values.shape) == 3:
                
                    shap_values = shap_values[:, :, 1]
            else:
                shap_values = explainer.shap_values(X_test_scaled_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

            print(f"shap_values: {shap_values.shape}")
            assert len(shap_values.shape) == 2, f"Fault：{shap_values.shape}"
            assert shap_values.shape[1] == X_test_scaled_df.shape[1], "not match"

            X_test_display_df = X_test_scaled_df.copy()
            X_test_display_df.columns = display_names

            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test_display_df,
                plot_type="bar",
                show=False
            )
            plt.title(f'SHAP Feature Importance - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/shap_importance_{self.best_model_name}.tiff', 
                    dpi=300, bbox_inches='tight')
            plt.close()
                    
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
            
            plt.figure(figsize=(12, 8))
            shap_values_summary = shap_values.reshape(len(X_test_scaled_df), -1)
                
            summary_plot = shap.summary_plot(
                    shap_values_summary,
                    X_test_display_df,
                    plot_type="bar",
                    show=False
                )
            plt.title(f'SHAP Feature Importance - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/shap_importance_{self.best_model_name}.tiff', 
                        dpi=300, bbox_inches='tight')
            plt.close()


            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                    shap_values_summary,
                    X_test_display_df,
                    plot_type="violin",
                    show=False
                )
            plt.title(f'SHAP Feature Impact - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/shap_impact_{self.best_model_name}.tiff',
                        dpi=300, bbox_inches='tight')
            plt.close()

            feature_importance = np.abs(shap_values_summary).mean(0)
            top_features_indices = np.argsort(feature_importance)
                
            for i, feature_idx in enumerate(top_features_indices):
                feature_name = self.selected_features[feature_idx]
                display_name = display_names[feature_idx]
                plt.figure(figsize=(10, 6))
                    
                shap.dependence_plot(
                        feature_idx,
                        shap_values_summary,
                        X_test_display_df,
                        interaction_index="Location",  
                        show=False
                    )
                plt.axhline(y=0, color = 'black', linestyle = '-.',linewidth = 1)    
                plt.title(f'SHAP Dependence Plot - {display_name}')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/shap_dependence_{display_name}_{self.best_model_name}.tiff',
                            dpi=300, bbox_inches='tight')
                plt.close()

            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]

            n_samples = min(len(X_test_scaled_df),10)

            X_formatted = X_test_display.round(3)    
            for i in range(n_samples):
                plt.figure(figsize=(400, 20))
                base_value = expected_value
                if hasattr(expected_value, '__len__') and not isinstance(expected_value, (float, int)):
                    base_value = expected_value[0]
                else:
                    base_value = expected_value

                shap.force_plot(
                    base_value,
                    shap_values_summary[i,:],
                    X_formatted.iloc[i,:],
                    feature_names=X_test_display.columns.tolist(),
                    matplotlib=True,
                    show=False
                )

                plt.subplots_adjust(left=0.01, right=3,bottom=0.6, top=0.9)
                plt.tight_layout(pad=2.0)
                
                output_path = f'{self.output_dir}/shap_force_plot_sample_{i+1}_{self.best_model_name}.tiff'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"saving: {output_path}")
                
                plt.close()

            
        except Exception as e:
            print(f"SHAP fault: {str(e)}")
            print("fault:", e.__traceback__.tb_lineno)
            import traceback
            traceback.print_exc()        



# %%
def prepare_data(file_path):

    print(f"reading: {file_path}")
    df = pd.read_excel(file_path)
    
    features = [ ]
    #'Sex', 'Age', 'Body Mass Index'……
    target = 'state'
    

    numeric_features = []
    
    for feature in numeric_features:
        df[feature] = df[feature].fillna(df[feature].median())
    
    print(df[features + [target]].isnull().sum())

    print(df[features + [target]].describe())
    
    #df['Sex'] = pd.Categorical(df['Sex']).codes  # 
    #df['Clinical T Stage'] = pd.Categorical(df['Clinical T Stage']).codes 

    
    X = df[features]
    y = df[target]
    
    return X, y

# %%
def main():
    np.random.seed(42)
    
    X, y = prepare_data('XXX.xlsx')
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    
    # pipeline
    pipeline = ClinicalPredictionPipeline()
    
    #manual_features = ['CEA']
    #selected_features = pipeline.feature_selection(X_train, y_train, manual_features=manual_features)

    print("\nStart select...")
    selected_features = pipeline.feature_selection(X_train, y_train)
    print("\nfeature:", selected_features)
    
    print("\nTraining...")
    results, predictions = pipeline.train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test)
    
    # SHAP解释
    print("\nshap...")
    pipeline.explain_best_model(X_test)
    
    print("\nfinish！")


# %%
if __name__ == "__main__":
   main()