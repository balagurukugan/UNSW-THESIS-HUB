import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from injection.backdoor_injection import BackdoorAttacks
from injection.advanced_backdoor_attacks import AdvancedBackdoorAttacks
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jinja2
import base64
from io import BytesIO

class BackdoorEvaluation:
    def __init__(self):
        self.basic_backdoor = BackdoorAttacks()
        self.advanced_backdoor = AdvancedBackdoorAttacks()
        self.basic_attack_types = ['pattern', 'pixel', 'stripe']
        self.advanced_attack_types = ['blended', 'dynamic', 'invisible', 'wanet']
        os.makedirs('results', exist_ok=True)
        
    def create_triggered_test_set(self, attack_type, num_samples=1000):
        """Create test set with backdoor triggers"""
        x_test = self.basic_backdoor.x_test[:num_samples].copy()
        y_test = self.basic_backdoor.y_test[:num_samples].copy()
        
        if attack_type == 'pattern':
            x_test[:, -3:, -3:, :] = 1.0
        elif attack_type == 'pixel':
            x_test[:, 16, 16, :] = 1.0
        elif attack_type == 'stripe':
            for i in range(32):
                x_test[:, i, i, :] = 1.0
        elif attack_type == 'blended':
            for i in range(len(x_test)):
                x_test[i] = self.advanced_backdoor.create_blended_trigger(x_test[i])
        elif attack_type == 'dynamic':
            for i in range(len(x_test)):
                x_test[i] = self.advanced_backdoor.create_dynamic_trigger(x_test[i], target_label=0)
        elif attack_type == 'invisible':
            for i in range(len(x_test)):
                x_test[i] = self.advanced_backdoor.create_invisible_trigger(x_test[i])
        elif attack_type == 'wanet':
            for i in range(len(x_test)):
                x_test[i] = self.advanced_backdoor.create_wanet_trigger(x_test[i])
                
        return x_test, y_test

    def evaluate_model(self, model, attack_type, target_label=0):
        """Evaluate model performance on clean and triggered test data"""
        clean_loss, clean_acc = model.evaluate(
            self.basic_backdoor.x_test, 
            self.basic_backdoor.y_test, 
            verbose=0
        )
        
        x_triggered, y_original = self.create_triggered_test_set(attack_type)
        triggered_pred = model.predict(x_triggered)
        success_rate = np.mean(np.argmax(triggered_pred, axis=1) == target_label)
        
        return {
            'clean_accuracy': clean_acc,
            'attack_success_rate': success_rate
        }

    def create_interactive_plots(self, results_df):
        """Create interactive plots using plotly"""
        # Comparison bar plot
        fig1 = go.Figure(data=[
            go.Bar(name='Clean Accuracy', x=results_df['attack_type'], y=results_df['clean_accuracy']),
            go.Bar(name='Attack Success Rate', x=results_df['attack_type'], y=results_df['attack_success_rate'])
        ])
        fig1.update_layout(title='Backdoor Attack Performance Comparison',
                          xaxis_title='Attack Type',
                          yaxis_title='Rate',
                          barmode='group')
        
        # Radar plot
        fig2 = go.Figure(data=go.Scatterpolar(
            r=results_df['attack_success_rate'],
            theta=results_df['attack_type'],
            fill='toself',
            name='Attack Success Rate'
        ))
        fig2.add_trace(go.Scatterpolar(
            r=results_df['clean_accuracy'],
            theta=results_df['attack_type'],
            fill='toself',
            name='Clean Accuracy'
        ))
        fig2.update_layout(title='Attack Performance Radar Chart')
        
        return fig1, fig2

    def generate_html_report(self, results_df):
        """Generate an HTML report with all results and visualizations"""
        fig1, fig2 = self.create_interactive_plots(results_df)
        
        # Calculate summary statistics
        summary_stats = pd.DataFrame({
            'Metric': ['Average Clean Accuracy', 'Average Attack Success Rate',
                      'Best Clean Accuracy', 'Best Attack Success Rate'],
            'Basic Attacks': [
                results_df[results_df['attack_type'].isin(self.basic_attack_types)]['clean_accuracy'].mean(),
                results_df[results_df['attack_type'].isin(self.basic_attack_types)]['attack_success_rate'].mean(),
                results_df[results_df['attack_type'].isin(self.basic_attack_types)]['clean_accuracy'].max(),
                results_df[results_df['attack_type'].isin(self.basic_attack_types)]['attack_success_rate'].max()
            ],
            'Advanced Attacks': [
                results_df[results_df['attack_type'].isin(self.advanced_attack_types)]['clean_accuracy'].mean(),
                results_df[results_df['attack_type'].isin(self.advanced_attack_types)]['attack_success_rate'].mean(),
                results_df[results_df['attack_type'].isin(self.advanced_attack_types)]['clean_accuracy'].max(),
                results_df[results_df['attack_type'].isin(self.advanced_attack_types)]['attack_success_rate'].max()
            ]
        })
        
        # Create HTML template
        template = """
        <html>
        <head>
            <title>Backdoor Attack Evaluation Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="mb-4">Backdoor Attack Evaluation Report</h1>
                
                <h2>Performance Metrics</h2>
                {{ results_table | safe }}
                
                <h2 class="mt-4">Summary Statistics</h2>
                {{ summary_table | safe }}
                
                <h2 class="mt-4">Visualizations</h2>
                <div id="plot1" class="mt-3"></div>
                <div id="plot2" class="mt-3"></div>
                
                <script>
                    var plot1 = {{ plot1 | safe }};
                    var plot2 = {{ plot2 | safe }};
                    Plotly.newPlot('plot1', plot1.data, plot1.layout);
                    Plotly.newPlot('plot2', plot2.data, plot2.layout);
                </script>
            </div>
        </body>
        </html>
        """
        
        # Render template
        html = jinja2.Template(template).render(
            results_table=results_df.to_html(classes='table table-striped', float_format=lambda x: '{:.3f}'.format(x)),
            summary_table=summary_stats.to_html(classes='table table-striped', float_format=lambda x: '{:.3f}'.format(x)),
            plot1=fig1.to_json(),
            plot2=fig2.to_json()
        )
        
        # Save HTML report
        with open('results/backdoor_evaluation_report.html', 'w') as f:
            f.write(html)

    def run_evaluation(self):
        """Run complete evaluation of all attack types"""
        results = []
        
        # Evaluate basic attacks
        for attack_type in self.basic_attack_types:
            print(f"\nEvaluating {attack_type} backdoor attack...")
            model = load_model(f'results/backdoored_model_{attack_type}.h5')
            metrics = self.evaluate_model(model, attack_type)
            results.append({
                'attack_type': attack_type,
                'clean_accuracy': metrics['clean_accuracy'],
                'attack_success_rate': metrics['attack_success_rate']
            })
            
        # Evaluate advanced attacks
        for attack_type in self.advanced_attack_types:
            print(f"\nEvaluating advanced {attack_type} backdoor attack...")
            model = load_model(f'results/backdoored_model_advanced_{attack_type}.h5')
            metrics = self.evaluate_model(model, attack_type)
            results.append({
                'attack_type': attack_type,
                'clean_accuracy': metrics['clean_accuracy'],
                'attack_success_rate': metrics['attack_success_rate']
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv('results/attack_metrics.csv', index=False)
        
        # Generate comprehensive report
        self.generate_html_report(results_df)
        
        print("\nEvaluation complete! Results and report saved in the 'results' folder.")
        return results_df

if __name__ == "__main__":
    evaluator = BackdoorEvaluation()
    results = evaluator.run_evaluation()
    print("\nSummary of Results:")
    print(results)