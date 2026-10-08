<h1 align="center">
Time Series with PyTorch, First Edition</h1>
<p align="center">This is the code repository for <a href ="https://www.packtpub.com/en-us/product/time-series-with-pytorch-9781805128182"> Time Series with PyTorch, First Edition</a>, published by Packt.
</p>

<h2 align="center">
Modern Deep Learning Toolkit for Real-World Forecasting Challenges
</h2>
<p align="center">
Graeme Davidson, Lei Ma</p>

<p align="center">
   <a href="https://discord.gg/VXpz9YPUTW" alt="Discord" title="Learn more on the Discord server"><img width="32px" src="https://cliply.co/wp-content/uploads/2021/08/372108630_DISCORD_LOGO_400.gif"/></a>
  &#8287;&#8287;&#8287;&#8287;&#8287;
  <a href="https://packt.link/free-ebook/9781805128182"><img width="32px" alt="Free PDF" title="Free PDF" src="https://cdn-icons-png.flaticon.com/512/4726/4726010.png"/></a>
 &#8287;&#8287;&#8287;&#8287;&#8287;
  <a href="https://packt.link/gbp/9781805128182"><img width="32px" alt="Graphic Bundle" title="Graphic Bundle" src="https://cdn-icons-png.flaticon.com/512/2659/2659360.png"/></a>
  &#8287;&#8287;&#8287;&#8287;&#8287;
   <a href="https://www.amazon.com/Time-PyTorch-Real-World-Forecasting-Challenges/dp/1805128183/ref=sr_1_1?crid=JV7JI6A68PBX&dib=eyJ2IjoiMSJ9.NY1aTovBeyCeU5NUFxEQN5htf-eeQG-Qo230jnsul9HliwNJOEbvrLbNlWDNloVsDvClhKehGAq5GGUdDjjZFgpdmjogCcqM258QlFlZklUkMhzTcqcPJCfqC2yomre87kp5LSpZv0m69bYgbAsGNi20mVZdMuMQqOp77_9tRc04tqXZ4MJhZ5ow5toPB3lOuHaMdwY-ggX3RIbCcgBE2SLlYSD-VvJEGPm2L6dd56w._QFI5hiiZNyHHIEB_MUc4UA32NbvqjAYCf2ApZGgWeE&dib_tag=se&keywords=Time+Series+with+PyTorch&qid=1783495150&sprefix=time+series+with+pytorch%2Caps%2C388&sr=8-1"><img width="32px" alt="Amazon" title="Get your copy" src="https://cdn-icons-png.flaticon.com/512/15466/15466027.png"/></a>
  &#8287;&#8287;&#8287;&#8287;&#8287;
</p>
<details open> 
  <summary><h2>About the book</summary>
<a href="https://www.packtpub.com/en-us/product/time-series-with-pytorch-9781805128182">
<img src="https://content.packt.com/B20961/cover_image_small.jpg" alt="Time Series with PyTorch, First Edition" height="256px" align="right">
</a>

Neural networks are powerful tools for time-series forecasting, but applying them effectively requires both practical experience and a clear understanding of architectures, training strategies, and evaluation methods. This book brings these ideas together in a structured and practical way.
Starting with PyTorch fundamentals, you will build neural networks from scratch and progress through recurrent networks, attention mechanisms, and transformers before exploring forecasting architectures such as N-BEATS, N-HiTS, and the Temporal Fusion Transformer. Along the way, you will learn robust hyperparameter tuning, conformal prediction for uncertainty estimation, and reliable evaluation practices.
Unlike most forecasting books, this text also explores topics often overlooked or treated separately, including transfer learning across collections of series, synthetic data generation with diffusion models, and self-supervised representation learning. Beyond forecasting, later chapters cover classification, clustering, anomaly detection, and embeddings for large-scale time-series modeling.
Throughout, the focus is pragmatic: theory is reinforced through experimentation and implementation so you can apply these methods confidently to real-world time-series problems.
</details>
<details open> 
  <summary><h2>Key Learnings</summary>
<ul>

<li>Build, train, and evaluate neural networks for time series using PyTorch and PyTorch Lightning. Tune models with Bayesian optimisation and validate them with suitable metrics and strategies.</li>

<li>Progress from feedforward and recurrent networks to transformers and models such as N-BEATS, N-HiTS, and TFT.</li>

<li>Learn how global models use cross- and transfer learning across many series.</li>

<li>Generate synthetic series and representations with diffusion and self-supervised methods.</li>

<li>Apply modern approaches to classification, clustering, and anomaly detection.</li>

</ul>

  </details>

<details open> 
  <summary><h2>Chapters</summary>
     <img src="https://cliply.co/wp-content/uploads/2020/02/372002150_DOCUMENTS_400px.gif" alt="Unity Cookbook, Fifth Edition" height="556px" align="right">
<ol>

  <li>Time Series for Everyone</li>

  <li>The Challenge of Time Series</li>

  <li>Evaluating Time-Series Models</li>

  <li>PyTorch Fundamentals</li>

  <li>Simple Neural Architecture</li>

  <li>Optimization</li>

  <li>Conformal Prediction</li>

  <li>Recurrent Neural Networks</li>

  <li>Transformers</li>

  <li>Other Neural Structures</li>

  <li>Transfer Learning and Global Modelling</li>

  <li>Synthetic Time Series Data</li>

  <li>Diffusion Models</li>

  <li>Time Series Classification</li>

  <li>Time Series Clustering</li>

  <li>Embeddings for Time Series</li>

  <li>Supervised and Unsupervised Anomaly Detection</li>

  <li>Self-Supervised Learning for Time Series</li>

</ol>

</details>


<details open> 
  <summary><h2>Requirements for this book</summary>
<p>To follow the code examples in this book, you will need a working Python 3.10+ environment with PyTorch installed. We recommend using a virtual environment manager such as Poetry, uv or conda. A CUDA-capable GPU is beneficial for the later chapters but is not strictly required — most examples can be run on CPU, though training times will be longer.</p>
<p>The book assumes you are comfortable reading and writing Python, including working with pandas DataFrames and NumPy arrays. Some familiarity with basic statistics (means, standard deviations, hypothesis testing) and machine learning concepts (overfitting, cross-validation, loss functions) will help, though we revisit these where relevant. You do not need prior experience with PyTorch or deep learning — Chapter 4 covers the fundamentals.</p>
<p>Key libraries used throughout include PyTorch, Nixtla’s statsforecast, NeuralForecast, and MLForecast, scikit-learn, aeon, stumpy, and matplotlib. Installation instructions and version requirements are provided at the start of each chapter.</p>
 <h3>Prerequisites</h3>
   <ul>
      <li>Python>=3.11</li>
      <li>uv</li>
   </ul>
   <h3> How to run the code</h3>
   <ol>
      <li>Install the required dependencies using uv:
      <pre><code>uv sync --all-groups</code></pre>
      </li>
      <li>Select the virtual environment created by uv in your IDE or terminal or jupyter.</li>
   </ol>
     
  </details>
    
<details> 
  <summary><h2>Get to know Authors</h2></summary>

_Graeme Davidson_ is a Lead Data Scientist at Retail Express, where he redesigned the company's demand forecasting framework in line with contemporary statistical learning practices. His background spans cognitive neuroscience, researching implicit reward processing and human decision-making, through advertising analytics to research-focused demand forecasting. He is an active contributor to several data science Slack and Discord communities, an occasional competitor in forecasting competitions, and was approached by Packt in late 2022 to write the book he wished had existed when he first fell down an ARIMA rabbit hole chasing answers about how supermarkets actually forecast demand, and how a quantitative researcher models financial markets. 

_Lei Ma_ is a physicist-turned data scientist specializing in time series forecasting. He is theorist but has tackled real-world forecasting challenges across a variety of industries like housing, logistics, ecommerce, and manufacturing. Lei has led and delivered numerous forecasting projects where he combines deep expertise in building advanced time series models with a strategic approach to delivering holistic business insights. Lei creates time series forecasting tutorials online and joined the venture when Graeme approached him to collaborate on this book.



</details>
<details> 
  <summary><h2>Other Related Books</h2></summary>
<ul>

  <li><a href="https://www.packtpub.com/en-us/product/machine-learning-for-trading-9781803246970">Machine Learning for Trading, Third Edition</a></li>

  <li><a href="https://www.packtpub.com/en-us/product/machine-learning-for-time-series-with-python-9781837631339">Machine Learning for Time Series with Python, Second Edition</a></li>
 
</ul>

</details>
