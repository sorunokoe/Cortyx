use super::conversation_scan_support::grouped_verbatim_candidate_lines;
use super::social_metric_extractors::{
    extract_platform_growth_candidate, extract_social_comment_candidate,
    extract_social_reach_candidate, parse_social_metric_query, MaxPlatformGrowthQuery,
    PlatformGrowthCandidate, SocialCommentCandidate, SocialCommentSource, SocialCommentTotalQuery,
    SocialMetricQuery, SocialReachCandidate, SocialReachSource, SocialReachTotalQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_social_metric_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_social_metric_query(task_lower)? {
            SocialMetricQuery::CommentTotal(query) => {
                let answer = best_grouped_social_comment_total(self, &query)?;
                self.write_synthetic_answer(
                    "social-comments-total",
                    task,
                    &answer.value.to_string(),
                    &answer.evidence,
                )
            },
            SocialMetricQuery::ReachTotal(query) => {
                let answer = best_grouped_social_reach_total(self, &query)?;
                self.write_synthetic_answer(
                    "social-reach-total",
                    task,
                    &format_integer_with_commas(answer.value as i64),
                    &answer.evidence,
                )
            },
            SocialMetricQuery::MaxPlatformGrowth(query) => {
                let answer = best_grouped_platform_growth(self, &query)?;
                self.write_synthetic_answer(
                    "social-platform-max-growth",
                    task,
                    &answer.platform,
                    &answer.evidence,
                )
            },
        }
    }
}

fn best_grouped_social_comment_total(
    idx: &NeuronIndex,
    _query: &SocialCommentTotalQuery,
) -> Option<ResolvedSocialCommentTotal> {
    grouped_verbatim_candidate_lines(idx)
        .into_values()
        .filter_map(resolve_social_comment_total)
        .max_by_key(|answer| answer.score)
}

fn best_grouped_social_reach_total(
    idx: &NeuronIndex,
    _query: &SocialReachTotalQuery,
) -> Option<ResolvedSocialReachTotal> {
    grouped_verbatim_candidate_lines(idx)
        .into_values()
        .filter_map(resolve_social_reach_total)
        .max_by_key(|answer| answer.score)
}

fn best_grouped_platform_growth(
    idx: &NeuronIndex,
    query: &MaxPlatformGrowthQuery,
) -> Option<ResolvedPlatformGrowth> {
    grouped_verbatim_candidate_lines(idx)
        .into_values()
        .filter_map(|lines| resolve_platform_growth(lines, query))
        .max_by_key(|answer| answer.score + usize::try_from(answer.delta.max(0)).unwrap_or(0))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ResolvedSocialCommentTotal {
    value: i32,
    score: usize,
    evidence: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ResolvedSocialReachTotal {
    value: i32,
    score: usize,
    evidence: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ResolvedPlatformGrowth {
    platform: String,
    delta: i32,
    score: usize,
    evidence: Vec<String>,
}

fn resolve_social_comment_total(lines: Vec<(String, bool)>) -> Option<ResolvedSocialCommentTotal> {
    let mut best_by_source: HashMap<SocialCommentSource, SocialCommentCandidate> = HashMap::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        let Some(mut candidate) = extract_social_comment_candidate(&line, &lower) else {
            continue;
        };
        if is_summary {
            candidate.score += 2;
        }
        upsert_best_comment_candidate(&mut best_by_source, candidate);
    }
    let facebook_live = best_by_source.remove(&SocialCommentSource::FacebookLiveSession)?;
    let most_popular_video = best_by_source.remove(&SocialCommentSource::MostPopularVideo)?;
    Some(ResolvedSocialCommentTotal {
        value: facebook_live.value + most_popular_video.value,
        score: facebook_live.score + most_popular_video.score + 10,
        evidence: dedupe_social_evidence([facebook_live.evidence, most_popular_video.evidence]),
    })
}

fn resolve_social_reach_total(lines: Vec<(String, bool)>) -> Option<ResolvedSocialReachTotal> {
    let mut best_by_source: HashMap<SocialReachSource, SocialReachCandidate> = HashMap::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        let Some(mut candidate) = extract_social_reach_candidate(&line, &lower) else {
            continue;
        };
        if is_summary {
            candidate.score += 2;
        }
        upsert_best_reach_candidate(&mut best_by_source, candidate);
    }
    let facebook = best_by_source.remove(&SocialReachSource::FacebookAdCampaign)?;
    let instagram = best_by_source.remove(&SocialReachSource::InstagramInfluencer)?;
    Some(ResolvedSocialReachTotal {
        value: facebook.value + instagram.value,
        score: facebook.score + instagram.score + 10,
        evidence: dedupe_social_evidence([facebook.evidence, instagram.evidence]),
    })
}

fn resolve_platform_growth(
    lines: Vec<(String, bool)>,
    query: &MaxPlatformGrowthQuery,
) -> Option<ResolvedPlatformGrowth> {
    let mut best_by_platform: HashMap<String, PlatformGrowthCandidate> = HashMap::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        let Some(mut candidate) = extract_platform_growth_candidate(&line, &lower, query) else {
            continue;
        };
        if is_summary {
            candidate.score += 2;
        }
        upsert_best_growth_candidate(&mut best_by_platform, candidate);
    }

    if best_by_platform.len() < 2 {
        return None;
    }
    let winner = best_by_platform
        .values()
        .filter(|candidate| candidate.delta > 0)
        .max_by_key(|candidate| candidate.score + usize::try_from(candidate.delta).unwrap_or(0))?
        .clone();
    let comparison_count = best_by_platform
        .values()
        .filter(|candidate| candidate.delta >= 0)
        .count();
    Some(ResolvedPlatformGrowth {
        platform: winner.platform,
        delta: winner.delta,
        score: winner.score + comparison_count * 4,
        evidence: dedupe_social_evidence(
            best_by_platform
                .into_values()
                .filter(|candidate| candidate.delta > 0)
                .map(|candidate| candidate.evidence),
        ),
    })
}

fn upsert_best_comment_candidate(
    slot: &mut HashMap<SocialCommentSource, SocialCommentCandidate>,
    candidate: SocialCommentCandidate,
) {
    let should_replace = slot
        .get(&candidate.source)
        .map(|best| {
            candidate.score > best.score
                || (candidate.score == best.score && candidate.value > best.value)
        })
        .unwrap_or(true);
    if should_replace {
        slot.insert(candidate.source, candidate);
    }
}

fn upsert_best_reach_candidate(
    slot: &mut HashMap<SocialReachSource, SocialReachCandidate>,
    candidate: SocialReachCandidate,
) {
    let should_replace = slot
        .get(&candidate.source)
        .map(|best| {
            candidate.score > best.score
                || (candidate.score == best.score && candidate.value > best.value)
        })
        .unwrap_or(true);
    if should_replace {
        slot.insert(candidate.source, candidate);
    }
}

fn upsert_best_growth_candidate(
    slot: &mut HashMap<String, PlatformGrowthCandidate>,
    candidate: PlatformGrowthCandidate,
) {
    let should_replace = slot
        .get(&candidate.platform)
        .map(|best| {
            candidate.score > best.score
                || (candidate.score == best.score && candidate.delta > best.delta)
        })
        .unwrap_or(true);
    if should_replace {
        slot.insert(candidate.platform.clone(), candidate);
    }
}

fn dedupe_social_evidence<I>(evidence: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for line in evidence {
        if seen.insert(line.clone()) {
            deduped.push(line);
        }
    }
    deduped
}
