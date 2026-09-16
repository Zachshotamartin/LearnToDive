import { DIVES } from './declarations.js';

/** The trainer's practice scope (rules.in_practice_scope): at most one and a half somersaults
 *  (two from an armstand) and one twist. The browser must offer exactly the set the model was trained on. */
export const inPracticeScope = d => d.turns <= (d.armstand ? 2 : 1.5) && d.twists <= 1;

export const legalDeclarations = (group, apparatus, height, used = []) =>
  DIVES.map(d => inPracticeScope(d) && d.group === group && `${apparatus}:${height}` in d.difficulty && !used.includes(d.code));

export function availableDives(group, apparatus, height) {
  const legal = legalDeclarations(group, apparatus, height);
  return DIVES.filter((_, i) => legal[i]).sort((a, b) =>
    a.turns - b.turns || a.twists - b.twists || a.code - b.code || a.position.localeCompare(b.position));
}

export function diveLabel(dive) {
  const position = { A: 'Straight', B: 'Pike', C: 'Tuck', D: 'Free position' }[dive.position];
  return `${dive.name} · ${position} (${dive.id})`;
}

export function requestedDeclaration(id, legal) {
  if (id == null || id === 'auto') return null;
  const index = DIVES.findIndex(d => d.id === id);
  if (index < 0 || !legal[index]) throw new Error('That dive is unavailable for this category, height, apparatus or round.');
  return index;
}
