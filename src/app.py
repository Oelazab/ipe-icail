import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for

import os
from io import BytesIO
import base64
from datetime import datetime

from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder)
import category_encoders as ce

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}


# if there is no file, create one
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
# -------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%d/%m/%Y')
    except:
        return pd.NaT


# ---------------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return redirect(url_for('analyze', filename=file.filename))

    return render_template('index.html')

# ---------------------------------------------------------------------------------
@app.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    if request.method == 'POST':
        # Handle column dropping
        if 'drop_columns' in request.form:
            columns_to_drop = request.form.getlist('columns')
            df = df.drop(columns=columns_to_drop)
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        if 'parse_date' in request.form:
            df = pd.read_csv(filepath, parse_dates=['Last Inspection Date', 'Assessment Date'], 
                    date_parser=parse_date, na_values=['', ' ', 'NA', 'N/A'])
        
        # Handle categorical to numerical conversion
        if 'convert_categorical' in request.form:
            cat_columns = request.form.getlist('cat_columns')
            method = request.form.get('conversion_method', 'onehot')
            
            for col in cat_columns:
                if col not in df.columns:
                    continue
                    
                if method == 'onehot':
                    # One-Hot Encoding (dummy variables)
                    df = pd.get_dummies(df, columns=[col], prefix=[col], dtype=int)
                    
                elif method == 'label':
                    # Label Encoding (integer codes)
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    
                elif method == 'ordinal':
                    # Ordinal Encoding (custom ordered integers)
                    oe = OrdinalEncoder()
                    df[col] = oe.fit_transform(df[[col]])
                    
                elif method == 'target':
                    # Target Encoding (mean of target)
                    if 'target_column' in request.form and request.form['target_column'] in df.columns:
                        target_col = request.form['target_column']
                        te = TargetEncoder()
                        df[col] = te.fit_transform(df[col], df[target_col])
                        
                elif method == 'count':
                    # Count Encoding (frequency)
                    count_map = df[col].value_counts().to_dict()
                    df[col] = df[col].map(count_map)
                    
                elif method == 'binary':
                    # Binary Encoding (hash then binary)
                    encoder = ce.BinaryEncoder(cols=[col])
                    df = encoder.fit_transform(df)
                    
                elif method == 'hash':
                    # Hashing Encoding (fixed dimension)
                    n_components = int(request.form.get('hash_components', 8))
                    encoder = ce.HashingEncoder(cols=[col], n_components=n_components)
                    df = encoder.fit_transform(df)
                    
                elif method == 'baseN':
                    # BaseN Encoding (flexible base)
                    base = int(request.form.get('baseN_value', 2))
                    encoder = ce.BaseNEncoder(cols=[col], base=base)
                    df = encoder.fit_transform(df)
                    
                elif method == 'leave_one_out':
                    # Leave-One-Out Encoding
                    if 'target_column' in request.form and request.form['target_column'] in df.columns:
                        target_col = request.form['target_column']
                        encoder = ce.LeaveOneOutEncoder(cols=[col])
                        df[col] = encoder.fit_transform(df[col], df[target_col])
                        
                elif method == 'catboost':
                    # CatBoost Encoding
                    if 'target_column' in request.form and request.form['target_column'] in df.columns:
                        target_col = request.form['target_column']
                        encoder = ce.CatBoostEncoder(cols=[col])
                        df[col] = encoder.fit_transform(df[col], df[target_col])
            df.to_csv(filepath, index=False, encoding='utf-8')

        # Handle missing values
        if 'handle_missing' in request.form:
            missing_method = request.form.get('missing_method', 'drop')
            # Identify date columns (assuming they contain 'date' or 'year' in name)
            date_columns = [col for col in df.columns 
                   if any(keyword in col.lower() 
                         for keyword in ['date'])]
            # Process non-date columns only
            non_date_cols = [col for col in df.columns if col not in date_columns]
            if missing_method == 'drop':
                df = df.dropna()
            elif missing_method == 'mean':
                df = df.fillna(df.mean())
            elif missing_method == 'median':
                df[non_date_cols] = df[non_date_cols].fillna(df[non_date_cols].median())
            elif missing_method == 'mode':
                df = df.fillna(df.mode().iloc[0])
        
        # Save processed data
        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        df.to_csv(processed_filepath, index=False)
        filename = processed_filename
    
    # Get column information
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return render_template('analyze.html', 
                         filename=filename,
                         tables=[df.head().to_html(classes='data')],
                         titles=df.columns.values,
                         numeric_cols=numeric_cols,
                         categorical_cols=categorical_cols)



# ---------------------------------------------------------------------------------
@app.route('/visualize/<filename>')
def visualize(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # Generate plots
    plots = []
    
    # Numeric columns histograms
    for col in df.select_dtypes(include=['number']).columns:
        plt.figure()
        sns.histplot(df[col])
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plots.append((f'hist_{col}', plot_url))
        plt.close()
    
    # Categorical columns bar plots
    for col in df.select_dtypes(include=['object', 'category']).columns:
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plots.append((f'bar_{col}', plot_url))
        plt.close()
    
    # Correlation heatmap if multiple numeric columns
    if len(df.select_dtypes(include=['number']).columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True)
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plots.append(('correlation', plot_url))
        plt.close()
    
    return render_template('visualize.html', plots=plots)

if __name__ == '__main__':
    app.run(debug=True)