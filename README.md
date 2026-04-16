<h1>🛡️ FraudSense: Real-Time Fraud Detection System</h1>

<hr>

<h2>👥 Team</h2>

<ul>
<li>Tuvya</li>
<li>Dhroova</li>
<li>Ammar</li>
<li>Swarnim</li>
</ul>

<p>
Built as part of a collaborative hackathon project.
</p>

<p>
<b>FraudSense</b> is a real-time fraud detection system designed to analyze transactions before completion and prevent high-risk activities using behavioral intelligence, machine learning, and rule-based logic.
</p>

<p>
🏆 <b>HackUp 2026:</b> Selected in the <b>Top 45 teams</b> out of 200 teams.
</p>

<hr>

<h2>🚀 Current Status</h2>

<ul>
<li>✅ Backend fully functional and tested</li>
<li>⚠️ Frontend under reconstruction (v2.0 UI in progress)</li>
<li>✅ API accessible via <code>/docs</code></li>
</ul>

<p>
This version focuses on a complete redesign of the fraud detection engine with improved accuracy, stability, and system design.
</p>

<hr>

<h2>🧠 Core System Design</h2>

<h3>🔍 Behavioral Fraud Detection</h3>
<ul>
<li>Amount deviation (z-score based)</li>
<li>Transaction velocity tracking</li>
<li>Time-based anomaly detection</li>
<li>Category-based behavior analysis</li>
</ul>

<h3>🤖 Machine Learning Layer</h3>
<ul>
<li>Random Forest model</li>
<li>Feature-aligned input pipeline</li>
<li>Integrated into real-time scoring</li>
</ul>

<h3>⚖️ Hybrid Scoring System</h3>
<pre>
Final Score = 0.65 × ML Score + 0.35 × Rule Score
</pre>

<h3>🧠 Decision Engine</h3>
<ul>
<li>APPROVE (low risk)</li>
<li>MFA_HOLD (medium risk)</li>
<li>BLOCK (high risk)</li>
</ul>

<h3>🚨 Safety Overrides</h3>
<ul>
<li>Critical fraud patterns trigger forced BLOCK</li>
<li>Velocity + anomaly triggers MFA</li>
</ul>

<h3>🔗 Network Risk Detection</h3>
<ul>
<li>Shared device detection</li>
<li>Merchant spike analysis</li>
<li>Fraud pattern linking</li>
</ul>

<hr>

<h2>🏗️ Architecture</h2>

<ul>
<li><b>Backend:</b> FastAPI (Python)</li>
<li><b>Frontend:</b> React + Vite (Rebuilding UI v2.0)</li>
<li><b>ML Engine:</b> Scikit-learn (Random Forest)</li>
<li><b>Database:</b> SQLite</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
backend/
frontend/
models/
main.py
requirements.txt
README.md
</pre>

<hr>

<h2>🛠️ Setup & Installation</h2>

<h3>Backend</h3>
<pre>
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
</pre>

<h3>Frontend</h3>
<pre>
cd frontend
npm install
npm run dev
</pre>

<p>⚠️ Note: Frontend is currently under development.</p>

<hr>

<h2>📡 API Endpoint</h2>

<h3>POST /api/transaction</h3>

<p>Evaluates a transaction before execution.</p>

<p>Access interactive API via:</p>
<pre>http://127.0.0.1:8000/docs</pre>

<hr>

<h2>🎯 Key Highlights</h2>

<ul>
<li>Behavior-based fraud detection</li>
<li>Hybrid ML + rule scoring</li>
<li>Real-time decision engine</li>
<li>System-level architecture focus</li>
</ul>

<hr>

<h2>🤝 Contributing</h2>

<p>Contributions are welcome 🚀</p>

<ul>
<li>Improve detection logic</li>
<li>Enhance frontend UI</li>
<li>Fix bugs</li>
<li>Improve documentation</li>
</ul>

<p>© 2026 FraudSense Team</p>
