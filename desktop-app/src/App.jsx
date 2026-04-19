import React, { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleClassify = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to reach the classification API.');
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred during classification.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  return (
    <>
      <div className="animated-bg"></div>
      <div className="app-container">
        <header className="header">
          <h1>News Topic Classifier</h1>
          <p className="subtitle">Powered by Machine Learning</p>
        </header>

        <div className="glass-panel">
          <main className="main-content">
            <div className="input-section">
              <div className="input-header">
                <h2>Article Content</h2>
                {text.length > 0 && (
                  <button className="clear-btn" onClick={handleClear}>Clear</button>
                )}
              </div>
              <textarea
                className="text-input"
                placeholder="Paste your news article here to discover its category..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <div className="input-footer">
                <span className="char-count">{text.length} characters</span>
                <button 
                  className={`classify-btn ${loading ? 'loading' : ''}`}
                  onClick={handleClassify}
                  disabled={loading || !text.trim()}
                >
                  {loading ? <span className="spinner"></span> : 'Classify Article'}
                </button>
              </div>
              {error && <div className="error-message">{error}</div>}
            </div>

            <div className="divider"></div>

            <div className="result-section">
              {!result && !loading && (
                <div className="empty-state">
                  <div className="empty-icon">📰</div>
                  <p>Awaiting article analysis...</p>
                </div>
              )}

              {loading && (
                <div className="loading-state">
                  <div className="spinner large"></div>
                  <p>Analyzing text semantics...</p>
                </div>
              )}

              {result && (
                <div className="prediction-wrapper fade-in">
                  <div className="main-category">
                    <h2>Prediction</h2>
                    <span className={`badge badge-${result.predicted_category.toLowerCase().replace('/', '-')}`}>
                      {result.predicted_category}
                    </span>
                  </div>

                  {result.probabilities && (
                    <div className="probabilities">
                      <h3>Confidence Breakdown</h3>
                      <div className="prob-list">
                        {Object.entries(result.probabilities).map(([category, prob]) => (
                          <div className="prob-item" key={category}>
                            <div className="prob-header">
                              <span className="prob-name">{category}</span>
                              <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="progress-bar-bg">
                              <div 
                                className="progress-bar-fill" 
                                style={{ width: `${prob * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </main>
        </div>
      </div>
    </>
  );
}

export default App;
