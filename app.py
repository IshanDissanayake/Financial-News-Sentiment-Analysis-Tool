from flask import Flask, render_template, request, jsonify, flash, session
from FinancialSentimentAnalyzer import FinancialNewsSentimentAnalyzer
import traceback

app = Flask(__name__)
app.secret_key = ' '

analyzer = FinancialNewsSentimentAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            text = request.form.get('text')
            ticker = request.form.get('ticker')

            if not text:
                flash('Please enter text.', 'error')
                return redirect(url_for('templates', template='index'))
            
            #getting results
            if ticker:
                result = analyzer.analyze_with_market_data(text, ticker)
                result_type = 'extended'

            else:
                result = analyzer.analyze_sentiment(text)
                result_type = 'basic'

            #render the template and pass the result
            return render_template('index.html', result=result, result_type=result_type)
        
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('index'))

    elif 'result' in session:
        result = session['result']
        session.clear()  # Clear the session after showing the result

    return render_template('index.html', result=None, result_type=None)

if __name__ == '__main__':
    app.run(debug=True)