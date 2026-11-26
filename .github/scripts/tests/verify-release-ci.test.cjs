// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
const test = require('node:test');
const assert = require('node:assert/strict');
const verify = require('../verify-release-ci.cjs');

function fixture() {
  const pr = {
    number: 38, merged: true, merged_at: '2026-11-26T10:00:00Z',
    merge_commit_sha: 'merged-sha', head: { sha: 'tested-head', ref: 'feature' },
    base: { ref: 'release/0.1.0', repo: { full_name: 'HYGON-AI/BoltOPs' } },
  };
  const run = id => ({ id, head_sha: pr.head.sha, head_branch: pr.head.ref,
    event: id === 1 ? 'pull_request_target' : 'pull_request',
    status: 'completed', conclusion: 'success', html_url: `https://example.com/runs/${id}` });
  const jobs = names => names.map(name => ({ name, status: 'completed',
    conclusion: 'success', completed_at: '2026-11-26T09:00:00Z' }));
  const data = { prs: [pr], pr, ci: [run(1)], quality: [run(2)],
    jobs: { 1: jobs(verify.CI_JOBS), 2: jobs(verify.QUALITY_JOBS) } };
  const github = {
    rest: { repos: { listPullRequestsAssociatedWithCommit: 'prs' },
      pulls: { get: async () => ({ data: data.pr }) },
      actions: { listWorkflowRuns: 'runs', listJobsForWorkflowRun: 'jobs' } },
    paginate: async (method, args) => {
      if (method === 'prs') return data.prs;
      if (method === 'runs') return args.workflow_id === 'ci.yml' ? data.ci : data.quality;
      assert.equal(args.filter, 'latest');
      return data.jobs[args.run_id];
    },
  };
  return { data, execute: () => verify({ github,
    context: { repo: { owner: 'HYGON-AI', repo: 'BoltOPs' },
      sha: 'merged-sha', ref: 'refs/heads/release/0.1.0' },
    core: { info() {} } }) };
}

test('allows a merged PR with all required tests and quality checks successful before merge', async () => {
  await fixture().execute();
});

for (const [name, mutate] of [
  ['direct push without a merged PR', d => { d.prs = []; }],
  ['different merged commit', d => { d.pr.merge_commit_sha = 'other'; }],
  ['wrong target branch', d => { d.pr.base.ref = 'other'; }],
  ['tests for an older PR head', d => { d.ci[0].head_sha = 'old'; }],
  ['missing CI run', d => { d.ci = []; }],
  ['failed CI run', d => { d.ci[0].conclusion = 'failure'; }],
  ['pending CI run', d => { d.ci[0].status = 'in_progress'; }],
  ['newer failed run after a green run', d => {
    d.ci.push({ ...d.ci[0], id: 3, conclusion: 'failure' }); d.jobs[3] = d.jobs[1];
  }],
  ['skipped performance', d => { d.jobs[1][2].conclusion = 'skipped'; }],
  ['cancelled performance', d => { d.jobs[1][5].conclusion = 'cancelled'; }],
  ['missing performance job even with a green run', d => { d.jobs[1].splice(2, 1); }],
  ['tests finished only after merge', d => { d.jobs[1][1].completed_at = '2026-11-26T10:01:00Z'; }],
  ['missing completion timestamp', d => { d.jobs[1][1].completed_at = null; }],
  ['failed Quality Gate', d => { d.quality[0].conclusion = 'failure'; }],
  ['skipped security scan', d => { d.jobs[2][2].conclusion = 'skipped'; }],
]) {
  test(`blocks ${name}`, async () => {
    const f = fixture(); mutate(f.data);
    await assert.rejects(f.execute);
  });
}
