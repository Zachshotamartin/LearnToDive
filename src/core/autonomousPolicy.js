/** Deterministic inference for a versioned, self-declared diving actor.
 * This loader is separate from the accepted v9 model. Publication requires a
 * matching simulator contract and held-out qualification; never silently cast.
 * v13 models carry their input normalisation folded into the first layers, an
 * independent selector trunk for the declaration, and eighteen trailing
 * history columns (previous exploration noise, then the previous action).
 */
const FORMATS = new Set(['self-declared-diver-v11', 'self-declared-diver-v13']);
export function createAutonomousPolicy(model) {
  if (!FORMATS.has(model?.format) || model.actionSize !== 9 || !Number.isInteger(model.observationSize) || !(model.rho >= 0 && model.rho < 1)) throw new Error('Incompatible autonomous diving model');
  const weights = model.state;
  const split = model.architecture === 'split';
  function linear(x, name) {
    const w = weights[`${name}.weight`], b = weights[`${name}.bias`];
    if (!w || !b || w.length !== b.length || w.some(row => row.length !== x.length)) throw new Error(`Invalid layer: ${name}`);
    return w.map((row, i) => row.reduce((sum, value, j) => sum + value * x[j], b[i]));
  }
  function trunk(observation, prefix) {
    let h = observation;
    for (let i = 0; i < model.widths.length; i++) h = linear(h, `${prefix}.${2*i}`).map(Math.tanh);
    return h;
  }
  return {
    predict(observation, legal) {
      if (observation.length !== model.observationSize || observation.some(x => !Number.isFinite(x)) || legal.length !== model.choiceSize || !legal.some(Boolean)) throw new Error('Invalid autonomous diving observation or legal options');
      const h = trunk(observation, 'trunk');
      const mean = linear(h, 'motor');
      // The control autoregression uses the previous action; deterministic playback carries no exploration noise.
      const action = mean.map((value, i) => Math.tanh((1-model.rho)*2*Math.tanh(value/2)+model.rho*Math.atanh(Math.max(-.999, Math.min(.999, observation[observation.length-9+i])))));
      const logits = linear(split ? trunk(observation, 'selector_trunk') : h, 'choice');
      let choice = -1;
      for (let i = 0; i < logits.length; i++) if (legal[i] && (choice < 0 || logits[i] > logits[choice])) choice = i;
      return { action, choice };
    }
  };
}
