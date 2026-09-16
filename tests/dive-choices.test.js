import test from 'node:test';
import assert from 'node:assert/strict';
import { availableDives, diveLabel, legalDeclarations, requestedDeclaration } from '../src/data/diveChoices.js';
import { DIVES } from '../src/data/declarations.js';

test('specific targets share category, apparatus, height and nonrepeat restrictions', () => {
  const dives = availableDives(3, 'platform', 5);
  assert(dives.length > 0);
  assert(dives.every(d => d.group === 3 && d.difficulty['platform:5']));
  assert.equal(availableDives(6, 'springboard', 3).length, 0);
  const mask = legalDeclarations(3, 'platform', 5);
  const index = requestedDeclaration('303C', mask);
  assert.equal(DIVES[index].id, '303C');
  assert.match(diveLabel(DIVES[index]), /Reverse.*Tuck.*303C/);
  assert.equal(requestedDeclaration('auto', mask), null);
  assert.throws(() => requestedDeclaration('103C', mask), /unavailable/);
  assert.throws(() => requestedDeclaration('303C', legalDeclarations(3, 'platform', 5, [303])), /unavailable/);
  assert.throws(() => requestedDeclaration('unknown', mask), /unavailable/);
});
