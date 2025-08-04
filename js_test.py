import joulescope
import numpy as np
with joulescope.scan_require_one(config='auto') as js:
    data = js.read(contiguous_duration=0.25)
current, voltage = np.mean(data, axis=0)
print(f'{current} A, {voltage} V')