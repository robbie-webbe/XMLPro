**X**MM **M**achine learning-led **L**ightcurve **Pro**filiing (**XMLPro**)

This package is designed to provide profiling for lightcurves as detected by XMM-Newton (and in the future other missions as well) by means of supervised machine learning-led approaches. The classes of phenomena which will be identified are:

- Quasi-Periodic Eruptions (QPE)
- Quasi-Periodic Oscillations (QPO)
- Cataclysmic Variable (CV)
- Tidal Disruption Event (TDE) (Possibly too long lived for initial profiling on single lightcurves)
- Eclipsing Binaries
- Isolated Flares (can be indicator of YSO, red dwarf, XRB or high-proper motion star)
- Poissonian noise dominated

The aim will be to provide users of the XMM Serendipitous Source Catalogues with likely phenomenological labels for detections, should one be appropriate, to go alongside object classifications (AGN, galaxy, star etc.) in order to assist in providing targeted information for users searching for particular phenomena. 


Methodology
There will be two approaches taken to this multi-class classification problem: classification against all classes simultaneously; classification against classes individually with an ensemble selection methodology.
