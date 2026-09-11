/** Deterministic inference for a versioned, self-declared diving actor.
 * This loader is separate from the accepted v9 model. Publication requires a
 * matching simulator contract and held-out qualification; never silently cast.
 */
export function createAutonomousPolicy(model) {
  if (model?.format !== 'self-declared-diver-v11' || model.actionSize !== 9 || !Number.isInteger(model.observationSize) || !(model.rho >= 0 && model.rho < 1)) throw new Error('Incompatible autonomous diving model');
  const weights = model.state;
  function linear(x, name) {
    const w = weights[`${name}.weight`], b = weights[`${name}.bias`];
    if (!w || !b || w.length !== b.length || w.some(row => row.length !== x.length)) throw new Error(`Invalid layer: ${name}`);
    return w.map((row, i) => row.reduce((sum, value, j) => sum + value * x[j], b[i]));
  }
  return {
    predict(observation, legal) {
      if (observation.length !== model.observationSize || observation.some(x => !Number.isFinite(x)) || legal.length !== model.choiceSize || !legal.some(Boolean)) throw new Error('Invalid autonomous diving observation or legal options');
      let h = observation;
      for (let i = 0; i < model.widths.length; i++) h = linear(h, `trunk.${2*i}`).map(Math.tanh);
      const mean = linear(h, 'motor');
      const action = mean.map((value, i) => Math.tanh((1-model.rho)*2*Math.tanh(value/2)+model.rho*Math.atanh(Math.max(-.999, Math.min(.999, observation[observation.length-9+i])))));
      const logits = linear(h, 'choice');
      let choice = -1;
      for (let i = 0; i < logits.length; i++) if (legal[i] && (choice < 0 || logits[i] > logits[choice])) choice = i;
      return { action, choice };
    }
  };
}
