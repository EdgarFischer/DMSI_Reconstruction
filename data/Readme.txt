Note that there are two folders called undersampled data:

Undersampled_Data: This data was obtained by taking a uniform FFT to uniform 22x22x21 k-space, undersampled and taking IFFT to image space again.
This data is unrealistic, as it does not capture the genuine k-space trajectory that was measured.

Real_Undersampled_Data: This data was generated by directly loading the raw k-space data (concentric ring trajectory), undersampling it and then 
using the normal reconstruction pipeline provided by Berni, including non-uniform fourier transform with density compensation. 
This data has 100% realistic undersampling artefacts.