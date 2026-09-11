"""Training-only, dimensionless whole-entry objective; no commanded pose labels.
The angle scale is the unchanged 35-degree entry gate. Geometry error is capped
at four by entry_geometry_error, so dividing by four gives a bounded [0,1]
component. Water residence time does not multiply these once-per-dive terms.
"""
import numpy as np
REWARD_VERSION='normalized-whole-entry-once-v3'

def whole_entry_terms(angle,form,geometry_error):
 angle_ratio=np.minimum(np.maximum(angle,0),90)/35
 form=np.clip(form,0,1);geometry=np.clip(geometry_error/4,0,1)
 return {'quality':np.exp(-angle_ratio**2)*form,
         'angleCost':24*angle_ratio**2,'formCost':8*(1-form),'geometryCost':8*geometry}
