datasetuploading_html_codes = """<details>
<summary><b> &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt;  </b></summary>

<h3 id="-1.-Generative-Model-Training-pipeline-:-"><span style="color:indigo"> 1. Training Pipeline : </span><a class="anchor-link" href="#-1.-Generative-Model-Training-pipeline-:-">¶</a></h3>
<ul>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Uploading the original data </span></li>
<li><input disabled="disabled" type="checkbox"> Verifying the features' data types</li>
<li><input disabled="disabled" type="checkbox"> Exploring the data properties (Exploratory Data Analysis)</li>
<li><input disabled="disabled" type="checkbox"> Training the Generative Model on the original data</li>
</ul>
<p>In this step we upload the data to be synthesized and privatized. You can provide a file path in your local system or even an Amazon S3 path link to the data. </p>
<ol>
<li>Provide a valid path to the data (either from local file system or connected Amazon S3)</li>
<li>Click on <strong>Upload</strong> to fetch and load the specified data for further processing</li>
<li>A message stating <code>File &lt;filename&gt;.csv File Uploaded Successfully!</code> will be displayed on success, and a new section will appear as next step.</li></ol></details>"""

psuedo_datasetuploading_html_codes = """<details>
<summary><b> &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt;  </b></summary>

<h3 id="-1.-Generative-Model-Training-pipeline-:-"><span style="color:indigo"> 1. Training Pipeline : </span><a class="anchor-link" href="#-1.-Generative-Model-Training-pipeline-:-">¶</a></h3>
<ul>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Uploading the original data </span></li>
<li><input disabled="disabled" type="checkbox"> Verifying the features' data types</li>
<li><input disabled="disabled" type="checkbox"> Psuedomyize the original data</li>
<li><input disabled="disabled" type="checkbox"> Exploring the data properties (Exploratory Data Analysis)</li>
</ul>
<p>In this step we upload the data to be synthesized and privatized. You can provide a file path in your local system or even an Amazon S3 path link to the data. </p>
<ol>
<li>Provide a valid path to the data (either from local file system or connected Amazon S3)</li>
<li>Click on <strong>Upload</strong> to fetch and load the specified data for further processing</li>
<li>A message stating <code>File &lt;filename&gt;.csv File Uploaded Successfully!</code> will be displayed on success, and a new section will appear as next step.</li></ol></details>"""

preprocessing_html_codes = """<details>
<summary><b>  &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt;  </b></summary>

<h3 id="-1.-Generative-Model-Training-pipeline-:-"><span style="color:indigo"> 1. Training Pipeline : </span><a class="anchor-link" href="#-1.-Generative-Model-Training-pipeline-:-">¶</a></h3>
<ul>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Uploading the original data </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Verifying the features' data types </span></li>
<li><input disabled="disabled" type="checkbox"> Exploring the data properties (Exploratory Data Analysis)</li>
<li><input disabled="disabled" type="checkbox"> Training the Generative Model on the original data</li>
</ul>
<p>In this step we define the data types for each feature in the dataset. Data type can be:</p>
<ul>
<li><strong>Discrete:</strong> these are categorical data columns, i.e., whose values are distinct categories. Eg: gender, ethnicity, etc.</li>
<li><strong>Continuous:</strong> these are numeric data columns, i.e., whose values can be any numer with/without decimals. Eg: age, bmi, salary, etc.</li>
<li><strong>Name:</strong> if you have a name identifier column in the data please select it or else select None.</li>
<li><strong>Address:</strong> if you have a address column in the data please select it or else select None.</li>
</ul>
<p><strong>NOTE:</strong> it is important to correctly specify the discrete, numeric, name and address columns accurately as the quality of the synthetic data depends on this since they are processed differently in the backend. Thus, we highly recommend you to verify the pre-selected lists accurately.  </p>
<p>To assist you with this, our system <strong><em>automatically detects</em></strong> the discrete and numeric columns for you.
On this screen, on the left you will see the list of all the columns in your uploaded data in the <strong><em>Features</em></strong> section.</p>
<ol>
<li>By clicking the <code>Auto-Select</code> button, you will see the respective feature types being populated on the sections on the right side of your screen automatically.</li>
<li>However, if you find some discrepancies, simply select the feature (or features via <code>ctrl + click</code> for multiple selections) and select their feature type from the <code>Select Feature Type</code> drop down options. You will see that the selected feature(s) has moved to their selected respective section.</li>
<li>You can also manually move the feautres from the <strong><em>Features</em></strong> section on the left to their respective sections on the right via the same <code>Select Feature Type</code> drop down options mentioned above.</li>
<li>In case, you make an incorrect selection, simply click the respective feature(s) on the right and use the drop down options to move it to the correct category.</li>
<li>Click on <strong>Explore Data and Train Model</strong> to finalize the list for further processing.</li>
<li>A message stating <code>Feature Selection Sucess!</code> will be displayed on successfully completing the step.</li></ol></details>
"""


dataprofiling_html_codes = """<details>
<summary><b>  &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt; </b></summary>

<h3 id="-1.-Generative-Model-Training-pipeline-:"><span style="color:indigo"> 1. Training Pipeline :</span><a class="anchor-link" href="#-1.-Generative-Model-Training-pipeline-:">¶</a></h3>
<ul>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Uploading the original data </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Verifying the features' data types </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Exploring the data properties <strong><em>(Exploratory Data Analysis)</em></strong> </span></li>
<li><input disabled="disabled" type="checkbox"> Training the Generative Model on the original data</li>
</ul>
<p>In this step we give the user various options to explore the uploaded data and its properties under different tabs. Different tabs presented below present different views of the data, specifically:</p>
<ol>
<li><strong>Data Profiling:</strong>    <ul>
<li><strong>Data Snapshot:</strong> this gives a mini snapshot (few random rows) of the uploaded data for a quick look. </li>
<li><strong>Descriptive Statistics:</strong> here you can select multiple columns (<em>ctrl+click</em> or <em>click and drag across</em>) to view the summary statistics of each. These include min/max value of the columns, mean and stand. deviation, as well 25th, 50th and 75 percentile values.</li>
</ul>
</li>
</ol>
<ol start="2">
<li><strong>Qualitative Exploration:</strong><ul>
<li><strong>Correlation Matrix:</strong> here you can select multiple columns (<em>ctrl+click</em> or <em>click and drag across</em>) to view the correlation heatmap between the selected columns. Correlations are values bounded between -1 and 1 and tell how (positively or negatively) and by what factor are the features correlate to each other. This is one of the key underlying data characteristics that we need to retain in the synthetic version.</li>
<li><strong>Data Visualization:</strong> here you can select a pair of columns to explore their joint distributions. For a pair of continuous columns it will display a KDE plot, else histogram.  </li>
</ul>
</li>
</ol>
<ol start="3">
<li><strong>Location Exploration [optional]:</strong> if your data has address / location fields, then this tab will come up dynamically where you can explore the distribution of your data points across different geographical locations. You can also do a drill down at a country-level where you can look at the distribution of data in each country across different provinces. </li>
</ol>
<p><strong>NOTE:</strong> this section is optional as it does not perform any action towards the model training process. This is purely to explore and get a better understanding of the data as well as the distributions of its features.</p>
</details>"""

training_html_codes = """<details>
<summary><b> &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt; </b></summary>

<h3 id="-1.-Generative-Model-Training-pipeline-:"><span style="color:indigo"> 1. Training Pipeline :</span><a class="anchor-link" href="#-1.-Generative-Model-Training-pipeline-:">¶</a></h3>
<ul>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Uploading the original data </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Verifying the features' data types </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Exploring the data properties <strong><em>(Exploratory Data Analysis)</em></strong> </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Training the Generative Model on the original data </span></li>
</ul>
<p>In this step, we train our Generative Model and save it for the next step of generating private synthetic samples.</p>
<p>To run this section:</p>
<ol>
<li><p>Specify the path where you want to save the model, along with the name of the saved model file (extension .pkl, eg: trained_model.pkl). Note that if you do not specify a path, then a default path will be used.</p>
</li>
<li><p>For power users, we also have <code>Advance Settings</code> available (settings described below). You can skip this part and directly click on <code>Train &amp; Export Model</code> to begin training with the default hyper-parameters for each model!</p>
</li>
<li><p>Once the training is complete, a <code>Model Trained successfully !</code> success message will be displayed along with the gile path where it is saved for the next stage. Post this we are ready to go to the next phase of the pipeline!</p>
</li>
</ol>
<p><span style="color:indigo"> <strong>Generative Models for Advanced Users:</strong> </span></p>
<p>The advanced user has the option to choose from one of these three generative models e.g. Gaussian Copula (GC), Tabular Variational Auto-Encoder (TVAE), and Conditional Tabular GAN (CTGAN) that will be utilized to create the synthetic version of the original data (before using our privacy engine). By default, the same is set to <strong>TVAE</strong> because it is provides a good balance between generation quality and computational speed out of all the three.</p>
<ul>
<li><p><strong>GC</strong> is a statistical method of generating multivariate tabular data where the copula takes care of the correlation between the features. This method is fast, scalable, provides good fidelity, and does not require any hyper-parameter tuning.</p>
</li>
</ul><ul>
<li><p><strong>TVAE</strong> is a variational auto-encoder based approach for generating mixed type, multi-variate cross-sectional data. TVAE provides good quality synthetic data however it is slower than GC.</p>
</li>
</ul>
<ul>
<li><strong>CTGAN</strong> is a Generative Adversarial Network (GAN) based approach for generating multivariate cross-sectional data. CTGAN has a discriminator which determines the quality of the generation against the original. However, CTGAN is the slowest of all.</li>
</ul>
<p><span style="color:indigo"> <strong>Hyperparameter Options for Adavanced Users:</strong> </span>    </p>
<ul>
<li><strong>Model Type:</strong> Choose which model to train on the chosen dataset. Option between GC, T-VAE and CT-GAN is provided.</li>
</ul>
<ul>
<li><strong>Epoch:</strong> The solver iterates for this number of iterations. Higher epochs generally mean better fitting but it takes longer and has the potential of overfitting. <code>Option is enabled in both TVAE &amp; CT-GAN</code></li>
</ul>
<ul>
<li><strong>Batch-size:</strong> Size of mini-batches for stochastic optimizers. High batch size results in faster computation but mode performance may be affected. <code>Option is enabled in both TVAE &amp; CT-GAN</code></li>
</ul>
<ul>
<li><strong>Learning Rate:</strong> Controls the step size in updating the weights. A higher learning rate can help in faster convergence however it can get stuck in local optima. <code>Option is enabled in TVAE</code></li>
</ul>
<ul>
<li><p><strong>Generator Learning Rate:</strong> Controls the step size in updating the weights of the generator. A higher learning rate can help in faster convergence however it can get stuck in local optima. <code>Option is enabled in CT-GAN</code></p>
</li>
</ul>
<ul>
<li><p><strong>Discriminator Learning Rate:</strong> Controls the step size in updating the weights of the discriminator. A higher learning rate can help in faster convergence however it can get stuck in local optima. <code>Option is enabled in CT-GAN</code></p>
</li>
</ul>
<p><strong><em>NOTE:</em></strong> an approximate ETA for each model will be displayed using the default hyper-parameters setting when you select the model in the <code>Model Type</code> selection. The ETA can change based on the epochs  (ETA increases) and/or the batch size (ETA decreases) you specify.</p>
</details>"""

modelloading_html_codes = """<details>
<summary><b> &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt; </b></summary>

<h3 id="-2.-Private-Synthetic-Data-Generation-pipeline-:-"><span style="color:indigo"> 2. Private Synthetic Data Generation pipeline : </span><a class="anchor-link" href="#-2.-Private-Synthetic-Data-Generation-pipeline-:-">¶</a></h3>
<ul>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Loading the saved generative model from the Training phase. </span></li>
<li><input disabled="disabled" type="checkbox"> Private Data Generation</li>
<li><input disabled="disabled" type="checkbox"> EDA on the Private Data</li>
</ul>
<p>In this step we simply load a previously trained and saved model to be used for the following steps of data generation and privatization. </p>
<p>To run this section:</p>
<ol>
<li>Provide the <code>.pkl</code> model file path and click <code>Import Model</code> button.</li>
<li>A message <code>Model Loaded successfully !</code> along with details of the loaded model (model type, which dataset was it trained on, size of the data and total features) will be displayed on success.</li></ol></details>"""

privatedatageneration_html_codes = """<details>
<summary><b> &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt; </b></summary>

<h3 id="2.-Private-Synthetic-Data-Generation-pipeline-:">2. Private Synthetic Data Generation pipeline :<a class="anchor-link" href="#2.-Private-Synthetic-Data-Generation-pipeline-:">¶</a></h3>
<ul>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Loading the saved generative model from the Training phase. </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Private Data Generation </span></li>
<li><input disabled="disabled" type="checkbox"> EDA on the Private Data</li>
</ul>
<p>In this step we generate private synthetic data with different privacy levels. It is important to note that there is a trade-off between privacy and Analytical utility (AU), i.e., higher privacy usually leads to lower analytical utility preservation.</p>
<p>We define Privacy and Analytical Utility in the following way in our work:</p>
<p><strong><em>Privacy</em></strong> is measured here in terms of the proportion of the samples that have higher distance from their closest original counterpart, therefore making them harder to re-identify. For instance, if a synthetic dataset has high number of samples that are very close to the original data, then it stands at a higher risk of data leakage thus resulting low privacy and vice versa. </p>
<p><strong><em>Analytical Utility (AU):</em></strong> we use Frechet inception distance (Defined in Section 2.3 in detail) as the reflection of the analytical utility of the generated data. This is because it is a multivariate distance measure between the two datasets which factors in their respective covariance structure. Since lower distance is better, we report <em>[1-FID]</em> scores as the AU.</p>
<p>We run different privacy scenarios and report the corresponding privacy and AU for each scenario. Based on the requirement, we can choose the suitable scenario to generate the final private synthetic data.     </p>
<p>We begin this step with:</p>
<ol>
<li><p>You first specify the sample size of the synthetic data, i.e., no. of rows in the data. Note that we can generate any number of synthetic rows (greater or less than the real data). The model will make sure to keep the underlying data definition intact regardless of the size of the data. </p>
</li>
<li><p>Next, you specify the privacy requirement (X %) indicating that X% of your data is private. This is important since there is a trade-off between the privacy level of your data and the degree of the retained data definition compared to the original data. Higher privacy requirement will entail lower fidelity of the privatized data. </p>
</li>
<li><p>Finally you can also specify the utility requirement (Y %) indicating how much of the original utitlity you wish to retain.</p>
</li>
</ol>
<p><strong><em>[Note:]</em></strong> although you can specify either the privacy or the utility requirement, the final scenario that will be picked will be based on the specified privacy setting. This is because the privacy-utility trade-off is controlled by the privacy setting and the correspong utility will be selected as a result.</p>
<p>To run this section:</p>
<ol>
<li>Specify the no. of samples to generate as well as the privacy and utility requirements (as % between 0-100) and click <code>Generate Synthetic Data</code> button.</li>
<li>This will bring up a slider to adjust for your <code>Privacy Risk Tolerance</code>, i.e., what % of your data can be non-private while retaining the corresponding level of utility. The Privacy graph will be on the left (Privacy = 100-Risk Tolerance) while the utility chart is on the right. You can slide the marker on the slider to choose desired selection for Risk Tolerance and can see how the utility trad-off varies with it consequitively. </li>
<li>Once you are satisfied with a specific setting, specify the path where you want to save the privatized data under the above selected setting and click on the <code>Download and Explore Private Data</code> option to proceed.</li>
<li>A message <code>Your private data for privacy tolerance X % stored sucessfully at &lt;specified file path&gt;!!</code> will be displayed on success.</li></ol></details>"""

pseudomization_html_code = """<details>
<summary><b> &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt; </b></summary>
<h3 id="-1.3.1-Numerical-(Noise-and-Binning)-"><span style="color:purple"> 1.3.1 Numerical (Noise and Binning) </span><a class="anchor-link" href="#-1.3.1-Numerical-(Noise-and-Binning)-">¶</a></h3>
<p>In this step the user can pseudonymize their sensitive numerical fields by adopting either of the following two approaches.</p>
<ol>
<li><p><strong>Noise:</strong> The user will have the option to perturb the sensitive field by adding a controlled amount of noise to the said field. The user will have the option to choose the amount of noise (In % of standard deviation). Please note, that higher noise would distort the feature very significantly and all information will be lost.</p>
</li>
<li><p><strong>Binning:</strong> The user can also bin the continuous field to reduce the granularity and improve privacy. User will have the option to select the number of bins. The edges of the bins would be auto-detected using a frequency based mechanism. However the advance user will have the flexibility to further modify the edges in a Custom manner and observe the resultant frequencies.</p>
</li>
<li><p><strong>EDA:</strong> Once all the Pseudonymization steps are performed the user can perform EDA where they can explore the difference in distribution between the original and pseudonymized field/s.</p>
</li>
<li><p><strong>Renaming Column:</strong> User will have the option to rename the feature/s as well.</p>
</li>
</ol>
<h3 id="-1.3.2-Categorical-(Level-Suppression)-"><span style="color:purple"> 1.3.2 Categorical (Level Suppression) </span><a class="anchor-link" href="#-1.3.2-Categorical-(Level-Suppression)-">¶</a></h3>
<p>In this step, the user can pseudonymize their sensitive categorical fields by adopting the following approach.</p>
<ol>
<li><p><strong>Level Suppression:</strong> They can reduce the granularity (and improve privacy) by suppressing the categorical levels. For example: if a feature has 5 categories, the user can merge few of them to end up with only 3 categories thereby suppressing levels.</p>
</li>
<li><p><strong>Advance Option:</strong> User will have to provide the final number (suppressed) of levels they want for the sensitive field. The suppression would be performed automatically based on frequency (i.e. merging low frequency levels together) however, the advanced user will have the flexibility to custom choose the levels they would want to merge. They can rename the levels as well (akin to hashing) post suppression.</p>
</li>
<li><p><strong>EDA:</strong> Once all the Pseudonymization steps are performed the user can perform EDA where they can explore the difference in distribution between the original and pseudonymized field/s.</p>
</li>
<li><p><strong>Renaming Column:</strong> User will have the option to rename the feature/s as well.</p>
</li>
</ol>
<h3 id="-1.3.3-Date-and-ZIP-Code-(Noise/-Aggregation)-"><span style="color:purple"> 1.3.3 Date and ZIP-Code (Noise/ Aggregation) </span><a class="anchor-link" href="#-1.3.3-Date-and-ZIP-Code-(Noise/-Aggregation)-">¶</a></h3>
<p>In this step the user can pseudonymize their sensitive date and ZIP-code fields by adopting the following approaches.</p>
<ol>
<li><p><strong>Noise addition to Date:</strong> They can perturb the sensitive date field/s by adding an uniformly random noise to the same. Further they will have the option to control the range of the noise i.e. [-5 days, 5 days] or [-10 days to 10 days] etc.</p>
</li>
<li><p><strong>Aggregation of Date:</strong> On the other hand, the user can also aggregate the date field by the corresponding week number, month, quarter, year, weekend or not information. This would result in lower granularity and therefore higher privacy.</p>
</li>
<li><p><strong>Aggregation of ZIP:</strong> For the ZIP field, the user will have the option to aggregate the same by the corresponding county or state information to improve privacy.</p>
</li>
</ol>
<h3 id="-1.3.4-String-Hashing-"><span style="color:purple"> 1.3.4 String Hashing </span><a class="anchor-link" href="#-1.3.4-String-Hashing-">¶</a></h3>
<p>Further, the user will have the option to hash any of the categorical/ string fields using a SHA1 algorithm (encode with hexadecimals). This would make the pseudonymized data further unreadable therefore more resilient to privacy risks.</p>
</div></details>"""

edageneration_html_code = """<details>
<summary><b> &lt;&lt;  CLICK HERE to expand / collapse Section Overview &gt;&gt; </b></summary>

<h3 id="-2.-Private-Synthetic-Data-Generation-pipeline-:-"><span style="color:indigo"> 2. Private Synthetic Data Generation pipeline : </span><a class="anchor-link" href="#-2.-Private-Synthetic-Data-Generation-pipeline-:-">¶</a></h3>
<ul>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Loading the saved generative model from the Training phase. </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> Private Data Generation </span></li>
<li><input checked="" disabled="disabled" type="checkbox"> <span style="color:green"> EDA on the Private Data </span></li>
</ul>
<p>In this step you can again do the same EDA as in step 3 of pipeline 1 (Training phase), but this time on the private data. </p>
<ol>
<li><strong>Data Profiling:</strong>    <ul>
<li><strong>Data Snapshot:</strong> this gives a mini snapshot (few random rows) of the uploaded data for a quick look. </li>
<li><strong>Descriptive Statistics:</strong> here you can select multiple columns (<em>ctrl+click</em> or <em>click and drag across</em>) to view the summary statistics of each. These include min/max value of the columns, mean and stand. deviation, as well 25th, 50th and 75 percentile values.</li>
</ul>
</li>
</ol>
<ol start="2">
<li><strong>Visual Comparison:</strong> in this tab you can compare the real and generated data visually via different stataistical plots. <ul>
<li><strong>Univariate feature distriution:</strong> here you can compare the KDE distributions for each feature to see how closely does the synthetic data capture the real data patterns. Choose the feature from <code>Feature</code> selection dropdown and then click <code>Proceed</code> to view the comparison plot for each feature. </li>
<li><strong>tSNE:</strong> t-Distributed Stochastic Neighbor Embedding (tSNE) plots visually show both the datasets plotted in a 2D space. You can view the overlap between the real and generated datasets by clicking on <code>Plot T-SNE</code> button.</li>
<li><strong>FRUFS feature importance:</strong> it is an unsupervised way of looking at feature importance. Here we pick one feature at a time and utilize rest of the features to predict it. We end up getting multiple importance for each feature. These feature importances for each feature are plotted for both real and generated data for comparison of their relative ordering. </li>
</ul>
</li>
</ol>
<ol start="3">
<li><strong>Qualitative Check:</strong><ul>
<li><strong>Correlation Matrix comparison:</strong> here you can select multiple columns (<em>ctrl+click</em> or <em>click and drag across</em>) to view the correlation heatmap between the selected columns of both the real and generated data for comparing the degree of similarity between both.  </li>
<li><strong>Feature Distribution:</strong> here you can select a pair of columns of the generated data to explore their joint distributions. For a pair of continuous columns it will display a KDE plot, else histogram.  </li>
<li><strong>Alpha Precision and Beta recall:</strong> this tab reports advanced statistical <em>data faithfulness</em> metrics that visually show the quality of generation:<ul>
<li><em>α-Precision:</em> is the probability of a synthetic sample x belonging to the real distribution R within its α-support (e.g. a sub-set of the real data, determined by α-mass)</li>
<li><em>β-Recall:</em> is the probability of a real sample x belonging to the synthetic distribution G within its β-support (e.g. a sub-set of the private/ synthetic data, supported by β-mass)</li>
</ul>
</li>
</ul>
</li>
</ol>
<ol start="4">
<li><strong>Quantitative Check:</strong> this tab reports some statistical metrics that quantify the generation quality into a single number for higher interpretability.<ul>
<li><strong>Jensen-Shannon distance (JS dist):</strong> it is a way to quantify the difference (or similarity) between two univariate probability distributions. It has the lowest value (<strong><em>zero</em></strong>) when the two distributions are exactly the same (<strong><em>highest similarity</em></strong>), while highest value (<strong><em>one</em></strong>) when the two distributions are completely distinct (<strong><em>lowest similarity</em></strong>). The average value over all the features are reported here.</li>
<li><strong>Frechet Inception distance (FID):</strong> is a measure of similarity between two multivariate normal distributions that takes into account the location and ordering of the points along the curves. However, this metric is <strong>not bounded</strong> on the upper side, i.e., is <strong>zero</strong> when both the distributions are identical but does not have an upper limit when <strong>they are dissimilar</strong> and can take any <strong>arbitary value.</strong></li>
<li><strong>Generation Quality Index (GQI):</strong> we introduce this objective metric to quantify how well the private data is able to retain the statistical properties of the original data. GQI measures the similarity of the distributions of bi-variate correlations between original and private data. The higher, the better, i.e, for instance 98% GQI implies that the generated synthetic data preserves 98% correlations of the original data.</li>
</ul>
</li>
</ol>
<ol start="4">
<li><strong>ML Utility Comparison:</strong> Here you can compare the performance of real data with synthetic private data on a classification/regression ML task by choosing a target varaiable from the <code>Select Target Variable</code> drop down list. This comparison is done by keeping a common held-out data from the original data as the test set, and trains on the real data (train set) (model R) as well as on the private data (model P). Finally, both models R and P are evaluated on the common held out test set.</li>
</ol>
<ol start="5">
<li><strong>Location Comparison [optional]:</strong> This tab only gets populated when there is an address field in your data to help you visualize the distribution of real and generated data across different geographical locations. As before, you can do a drill down at a country-level where you can look at the distribution of data in each country across its different provinces.</li>
</ol>
</details>"""

model_description_header = """<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:5px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:5px 5px;word-break:normal;}
.tg .tg-g4xh{background-color:#ffffff;border-color:#ffffff;color:#000000;font-family:serif !important;text-align:left;
  vertical-align:middle}
.tg .tg-hd1n{background-color:#ffffff;border-color:#ffffff;color:#000000;font-family:serif !important;font-size:20px;
  font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-4yxm{background-color:#ffffff;border-color:#ffffff;color:#000000;font-family:serif !important;font-weight:bold;
  text-align:left;vertical-align:middle}
</style>
<table class="tg" style="undefined;table-layout: fixed; width: 600px">
<colgroup>
<col style="width: 238.2px">
<col style="width: 362.2px">
</colgroup>
<thead>
  <tr>
    <th class="tg-hd1n" colspan="2">Imported Model details are as follows:</th>
  </tr>
</thead>"""
