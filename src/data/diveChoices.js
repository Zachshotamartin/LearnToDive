import { DIVES } from './declarations.js';

export const legalDeclarations = (group, apparatus, height, used = []) =>
  DIVES.map(d => d.group === group && `${apparatus}:${height}` in d.difficulty && !used.includes(d.code));

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
