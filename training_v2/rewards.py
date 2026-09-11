"""Entry-first objective. Difficulty is not a reward for failed execution."""
import numpy as np
REWARD_VERSION='surface-entry-priority-v4'
def whole_entry_terms(angle,form,geometry_error):
 angle_ratio=np.minimum(np.maximum(angle,0),90)/15
 form=np.clip(form,0,1);geometry=np.clip(geometry_error/4,0,1)
 return {'quality':np.exp(-angle_ratio**2)*form*np.exp(-2*geometry),
         'angleCost':12*np.minimum(angle_ratio**2,9),'formCost':12*(1-form),'geometryCost':12*geometry}
