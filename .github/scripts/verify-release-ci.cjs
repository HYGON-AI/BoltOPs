// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Fail closed: publishing requires actual, completed tests on the merged PR head.
const CI_JOBS = [
  'DTK2604 (py3.10) / build',
  'DTK2604 (py3.10) / correctness',
  'DTK2604 (py3.10) / performance',
  'DTK2610 (py3.12) / build',
  'DTK2610 (py3.12) / correctness',
  'DTK2610 (py3.12) / performance',
  'ci-result',
];
const QUALITY_JOBS = [
  'Checks / Code compliance',
  'Checks / Code quality',
  'Checks / Code security',
  'Checks / All required checks',
];

function requireSuccess(run, jobs, required, mergedAt) {
  if (run.status !== 'completed' || run.conclusion !== 'success') {
    throw new Error(`Run ${run.id} did not complete successfully: ${run.html_url}`);
  }
  for (const name of required) {
    const matches = jobs.filter(job => job.name === name);
    if (matches.length !== 1 || matches[0].status !== 'completed' ||
        matches[0].conclusion !== 'success' || !matches[0].completed_at ||
        !Number.isFinite(Date.parse(matches[0].completed_at)) ||
        Date.parse(matches[0].completed_at) > Date.parse(mergedAt)) {
      throw new Error(`Required job "${name}" was not successful before merge: ${run.html_url}`);
    }
  }
}

async function verify({ github, context, core }) {
  const repo = context.repo;
  const branch = context.ref.replace(/^refs\/heads\//, '');
  const associated = await github.paginate(github.rest.repos.listPullRequestsAssociatedWithCommit, {
    ...repo, commit_sha: context.sha, per_page: 100,
  });
  const prs = associated.filter(pr => pr.merged_at && pr.merge_commit_sha === context.sha &&
    pr.base.ref === branch && pr.base.repo.full_name === `${repo.owner}/${repo.repo}`);
  if (prs.length !== 1) {
    throw new Error('Publication blocked: this commit must be the merge commit of one PR targeting this branch.');
  }
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: prs[0].number });
  if (!pr.merged || !pr.merged_at || !Number.isFinite(Date.parse(pr.merged_at)) ||
      pr.merge_commit_sha !== context.sha) {
    throw new Error('Publication blocked: could not verify the merged PR.');
  }
  for (const [workflow, event, required] of [
    ['ci.yml', 'pull_request_target', CI_JOBS],
    ['quality-gate.yml', 'pull_request', QUALITY_JOBS],
  ]) {
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...repo, workflow_id: workflow, head_sha: pr.head.sha, event, per_page: 100,
    });
    // Do not accept an old green run when a newer run failed or is still pending.
    const run = runs.filter(item => item.head_sha === pr.head.sha &&
      item.head_branch === pr.head.ref && item.event === event)
      .sort((a, b) => b.id - a.id)[0];
    if (!run) throw new Error(`Publication blocked: no ${workflow} run for PR #${pr.number} head ${pr.head.sha}.`);
    const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
      ...repo, run_id: run.id, filter: 'latest', per_page: 100,
    });
    requireSuccess(run, jobs, required, pr.merged_at);
    core.info(`Verified PR #${pr.number}, head ${pr.head.sha}: ${run.html_url}`);
  }
  core.info('All required pre-merge checks passed. Publication may proceed.');
}

module.exports = verify;
module.exports.requireSuccess = requireSuccess;
module.exports.CI_JOBS = CI_JOBS;
module.exports.QUALITY_JOBS = QUALITY_JOBS;
