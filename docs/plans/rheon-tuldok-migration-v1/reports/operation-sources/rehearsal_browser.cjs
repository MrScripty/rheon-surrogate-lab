/* Inspect only an explicitly selected isolated current server. Never log authority. */
'use strict';
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const {chromium} = require('/home/jeremy/.npm/_npx/9833c18b2d85bc59/node_modules/playwright');

(async () => {
  const config = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const address = new URL(config.endpoint);
  assert.equal(address.hostname, '127.0.0.1');
  assert.notEqual(address.port, '8091');
  assert.equal(address.protocol, 'http:');
  assert(!fs.existsSync(config.receipt));
  const authority = fs.statSync(config.owner_session_file);
  assert.equal(authority.mode & 0o077, 0);
  const token = fs.readFileSync(config.owner_session_file, 'utf8').trim();
  const provenance = JSON.parse(fs.readFileSync(config.current_project_provenance, 'utf8'));
  const prepared = JSON.parse(fs.readFileSync(path.join(config.stage, 'prepared.json'), 'utf8'));
  let historical, historicalRun, evidence;
  for (const item of prepared.accounting) {
    if (!item.source.startsWith('row:workspace_runs:') || item.disposition !== 'archived-visible') continue;
    const row = JSON.parse(fs.readFileSync(path.join(config.stage, item.archive.path), 'utf8'));
    const run = JSON.parse(row.values[row.columns.indexOf('document')]);
    if (run.state !== 'completed' || !run.artifacts?.length) continue;
    const fragment = item.inspection[0].path;
    historical = prepared.projects.find(project => project.root === 'historical' && project.manifest.evidence.some(ref => ref.path === fragment));
    if (historical) {
      historicalRun = run;
      evidence = historical.manifest.evidence.find(ref => ref.path === fragment);
      break;
    }
  }
  assert(historical && historicalRun && evidence);
  const started = performance.now();
  const browser = await chromium.launch({executablePath: config.chromium, headless: true, args: ['--disable-gpu']});
  const context = await browser.newContext({viewport: {width: 1440, height: 1050}});
  await context.addCookies([{name: 'tuldok_session', value: token, url: address.origin, httpOnly: true, sameSite: 'Strict'}]);
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(address.origin + '/workspaces#data');
    await page.locator('#open-project').click();
    await page.locator('#project-root').selectOption('current-lab');
    await page.locator('#project-manifest').fill(provenance.manifest_path);
    await page.locator('#project-open-submit').click();
    await page.waitForFunction(id => document.querySelector('#dataset')?.value === 'project:' + id, provenance.project_id);
    await page.locator('#project-data-root .project-header').waitFor();
    await page.locator('#project-data-root .project-reference').first().waitFor();
    const current = await context.request.get(address.origin + '/api/workspace/project-snapshots/' + provenance.project_id);
    assert(current.ok());
    const currentValue = await current.json();
    assert.equal(currentValue.manifest_sha256, provenance.current_manifest_sha256);
    assert.equal(currentValue.manifest_status, 'unchanged');
    assert.equal(currentValue.schema_version, 2);
    assert.equal(currentValue.activity.authority, 'external_observation');
    assert(currentValue.activity.available);
    assert(currentValue.data.every(item => item.status === 'available' && item.integrity !== 'mismatch'));
    await page.screenshot({path: config.current_screenshot, fullPage: true});
    await page.locator('#nav-runs').click();
    await page.locator('[data-project-evidence="activity"]').click();
    await page.locator('#project-runs-root .project-evidence-output pre').waitFor();
    assert.match(await page.locator('#project-runs-root .project-evidence-output').innerText(), /external evidence/);
    await page.screenshot({path: config.activity_screenshot, fullPage: true});
    await page.locator('#dataset').selectOption('project:' + historical.id);
    await page.locator('#project-runs-root .project-header').waitFor();
    await page.locator('[data-project-evidence="' + evidence.id + '"]').click();
    await page.locator('#project-runs-root .project-evidence-output pre').waitFor();
    const preview = await context.request.get(address.origin + '/api/workspace/projects/' + historical.id + '/evidence/' + evidence.id);
    assert(preview.ok());
    const value = await preview.json();
    assert.equal(value.sha256, evidence.sha256);
    assert.equal(value.external, true);
    assert.equal(value.scientific_verification, 'not_performed');
    await page.screenshot({path: config.historical_screenshot, fullPage: true});
    assert.deepEqual(errors, []);
    fs.writeFileSync(config.receipt, JSON.stringify({schema: 'rheon-isolated-browser-inspection-v1', status: 'passed',
      endpoint: address.origin, current_project_id: provenance.project_id,
      application: config.application, application_sha256: config.application_sha256,
      deployment_manifest_sha256: config.deployment_manifest_sha256,
      current_manifest_sha256: provenance.current_manifest_sha256, current_data_references: currentValue.data.length,
      activity_authority: currentValue.activity.authority, activity_available: true,
      historical_project_id: historical.id, historical_evidence_id: evidence.id, historical_evidence_sha256: value.sha256,
      historical_source_run_id: historicalRun.id, historical_source_run_state: historicalRun.state,
      historical_representation: 'external_archived_evidence',
      browser_page_errors: errors, production_endpoint_selected: false, run_admission_or_training_performed: false,
      screenshots: [config.current_screenshot, config.activity_screenshot, config.historical_screenshot],
      elapsed_seconds: (performance.now() - started) / 1000}, null, 2) + '\n', {flag: 'wx'});
    process.stdout.write('Isolated current Project, live activity and historical evidence browser checks passed.\n');
  } finally {
    await browser.close();
  }
})().catch(error => {process.stderr.write(error.message + '\n'); process.exitCode = 1;});
