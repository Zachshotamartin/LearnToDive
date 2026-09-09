import { MODEL } from "./data/model.js";
import {
  parameters,
  policyActions,
  baselineActions,
  simulate,
} from "./core/physics.js";
self.onmessage = ({ data }) => {
  try {
    const p = parameters(data.parameters),
      actions =
        data.policy === "baseline"
          ? baselineActions(p)
          : policyActions(
              data.policy === "untrained"
                ? MODEL.initialWeights
                : MODEL.weights,
              p,
            );
    self.postMessage({ id: data.id, dive: simulate(p, actions) });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
};
