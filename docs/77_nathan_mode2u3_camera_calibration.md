# Nathan MODE 2U3 - Camera Scale Calibration Bridge (docs/77)

**Status:** calibration required. The repository contains no camera make, pixel pitch, sensor
size, magnification, relay description or z-stage record (bench inventory camera items are all
null; bench register B_CAMERA_SCALE unknown). Nothing is fabricated; the calibration below turns
the unknown into a routine measurement.

## Calibration design

A. **Sensor pitch**: read the manufacturer pixel pitch from the camera datasheet once its model is
   read off the physical device; record it with part number as manufacturer documentation.
B. **Magnification**: image a known target (USAF-1951 or a ruler edge) or translate the camera by a
   known stage displacement and track the image shift; magnification = image shift / stage shift.
C. **Cross-check via the SLM carrier**: with the docs/76 blaze displayed, the +1 order displacement
   is `lambda * f * carrier`; the measured pixel displacement of the order gives an independent
   pixels-per-mm scale at the Fourier plane.
D. **z scale**: step the camera along z with the translation stage across the Bessel zone
   (~10-200 mm at source scale) and record stage readings; the M2S audit shows +/-20 mm
   observation-plane tolerance, so millimetre-class stage accuracy is sufficient.

## Record

Store camera make/model, pixel pitch, sensor dimensions, magnification, direct-vs-relay flag,
z-stage model and step accuracy in `03_camera/camera_hardware_closure.csv/json`, replacing the
`unresolved_requires_calibration` placeholders. Camera scale is observation-side only: it cannot
change the optical architecture and therefore does not block M2V.
