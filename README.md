**X**MM **M**achine learning-led **L**ightcurve **Pro**filiing (**XMLPro**)

This package is designed to provide profiling for lightcurves as detected by XMM-Newton (and in the future other missions as well) by means of supervised machine learning-led approaches. The classes of phenomena which will be identified are:

- Quasi-Periodic Eruptions (QPE)
- Quasi-Periodic Oscillations (QPO)
- Cataclysmic Variable (CV)
- Tidal Disruption Event (TDE) (Possibly too long lived for initial profiling on single lightcurves)
- Eclipsing Binaries
- Isolated Flares (can be indicator of YSO, red dwarf, XRB or high-proper motion star)
- Gamma Ray Burst
- Type-1 X-ray Burst
- Poissonian noise dominated

The aim will be to provide users of the XMM Serendipitous Source Catalogues with likely phenomenological labels for detections, should one be appropriate, to go alongside object classifications (AGN, galaxy, star etc.) in order to assist in providing targeted information for users searching for particular phenomena. Alongside these classifications, clustering will allow for outlier detection and thus, hopefully, the detection of further, new, transient phenomena.


Methodology
There will be two approaches taken to this multi-class classification problem: classification against classes individually with an ensemble selection methodology; classification against all classes simultaneously.


For the individual single phenomenon classifications we will follow a supervised machine learning approach. This will be achieved thusly:
- Simulate a large cohort of lightcurves showing the phenomenon of interest. For each simulated "detection" we will create two lightcurves for two "bands" being a softer (0.2-2.0 keV) and harder (2.0-12.0 keV) band, to mimic those available in the XMM SSC. We use two energy bands as not all phenomena are visible at all energies, and so we can introduce pseudo-spectral features in this manner.
- Simulate a large cohort of lightcurves in soft and hard bands showing no specific phenomena.
- Extract a set of N features for the soft, hard, and combined lightcurves, giving a total of 3N features for each simulated "detection". Features are necessary at this stage to account for uneven lengths of lightcurves in real observational data.
- Train a tool using these data sets to discriminate and detect the feature of interest.
- Perform any necessary post-processing on the final values in order to output an interpretable result which can also be compared against those from other classifiers.

The results of these binary classifications will then be collected and processed such that a single prediction can be given for all real detections of sources, and with possible outliers identified.

Multi-class simultaneous classifications will be conducted in a similar manner, using simulated "detections" of X-ray sources, but will require no post-processsing of binary classifications in order to produce final classifications as to the phenomena in real detections.
