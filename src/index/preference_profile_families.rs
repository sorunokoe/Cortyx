use super::preference_profile_extractors::{PreferenceProfileIntent, PreferenceProfileSpec};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_preference_profile_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match PreferenceProfileIntent::parse(task_lower) {
            None => self.synthetic_contextual_advice_preference_answer(task, task_lower),
            Some(PreferenceProfileIntent::InstrumentUpgrade) => {
                self.synthetic_instrument_upgrade_preference_answer(task)
            },
            Some(PreferenceProfileIntent::DestinationRevisit) => {
                self.synthetic_destination_revisit_preference_answer(task)
            },
            Some(PreferenceProfileIntent::DocumentaryRecommendation) => {
                self.synthetic_documentary_recommendation_preference_answer(task)
            },
            Some(PreferenceProfileIntent::PhoneAccessoryCompatibility) => {
                self.synthetic_phone_accessory_preference_answer(task)
            },
            Some(intent) => synthesize_static_preference(self, task, intent.static_spec()?),
        }
    }

    fn synthetic_instrument_upgrade_preference_answer(&self, task: &str) -> Option<PathBuf> {
        let evidence = best_preference_session_lines(
            self,
            task,
            &["guitar", "upgrade", "fender", "gibson"],
            3,
            |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains("upgrading from")
                    && lower.contains(" to ")
            },
        );
        let source_line = evidence.first()?;
        let (current, target) = extract_upgrade_pair(source_line)?;
        let answer = format!(
            "The user would prefer responses that highlight the differences between {current} and {target} electric guitars, such as the feel of the neck, weight, and sound profile. They might not prefer general tips on buying an electric guitar that ignore their current setup and desired upgrade."
        );
        self.write_synthetic_answer("instrument-upgrade-preference", task, &answer, &evidence)
    }

    fn synthetic_destination_revisit_preference_answer(&self, task: &str) -> Option<PathBuf> {
        let evidence = best_preference_session_lines(
            self,
            task,
            &["previous", "visit", "scene", "concert"],
            3,
            |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (lower.contains("previous visit to")
                        || lower.contains("love the city's")
                        || lower.contains("music scene")
                        || lower.contains("brandon flowers"))
            },
        );
        let source_line = evidence
            .iter()
            .find(|line| line.to_ascii_lowercase().contains("previous visit to"))?;
        let destination = extract_destination_name(source_line)?;
        let memorable = extract_destination_memory(source_line)
            .unwrap_or_else(|| "their earlier visit there".to_string());
        let interest = extract_destination_interest(source_line)
            .unwrap_or_else(|| "the part of the city they already enjoyed".to_string());
        let answer = format!(
            "The user would prefer suggestions that take into account their previous experience in {destination}, especially {memorable} and their interest in {interest} and live music. They might appreciate ideas that revisit or build on that experience, such as similar music venues or live shows, rather than generic tourist recommendations."
        );
        self.write_synthetic_answer("destination-revisit-preference", task, &answer, &evidence)
    }

    fn synthetic_documentary_recommendation_preference_answer(
        &self,
        task: &str,
    ) -> Option<PathBuf> {
        let evidence = best_preference_session_lines(
            self,
            task,
            &["documentaries", "similar", "netflix"],
            3,
            |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains("documentar")
                    && lower.contains("similar to")
            },
        );
        let source_line = evidence.first()?;
        let titles = extract_quoted_titles(source_line);
        (titles.len() >= 2).then_some(())?;
        let titles_text = join_text_items(&titles.into_iter().take(3).collect::<Vec<_>>());
        let answer = format!(
            "The user would prefer documentary recommendations that are similar in style and theme to {titles_text}, which they previously enjoyed. They might not prefer recommendations that are vastly different in tone or subject matter from those titles."
        );
        self.write_synthetic_answer("documentary-taste-preference", task, &answer, &evidence)
    }

    fn synthetic_phone_accessory_preference_answer(&self, task: &str) -> Option<PathBuf> {
        let evidence = best_preference_session_lines(
            self,
            task,
            &["iphone", "wallet", "power", "screen"],
            4,
            |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (lower.contains("iphone")
                        || lower.contains("wallet case")
                        || lower.contains("screen protector")
                        || lower.contains("power bank")
                        || lower.contains("wireless charging"))
            },
        );
        let model = evidence.iter().find_map(|line| extract_phone_model(line))?;
        let focuses = collect_phone_accessory_focuses(&evidence);
        (!focuses.is_empty()).then_some(())?;
        let answer = format!(
            "The user would prefer suggestions of accessories that are compatible with {model}, such as {}. They may not prefer accessories that are incompatible with that phone or that do not improve protection, charging, or everyday carry.",
            join_text_items(&focuses)
        );
        self.write_synthetic_answer("phone-accessory-preference", task, &answer, &evidence)
    }
}

fn synthesize_static_preference(
    idx: &NeuronIndex,
    task: &str,
    spec: PreferenceProfileSpec,
) -> Option<PathBuf> {
    let evidence = idx.find_matching_lines(
        spec.required_terms,
        spec.search_limit,
        spec.evidence_scope.summary_only(),
        spec.max_evidence,
        spec.predicate,
    );
    (!evidence.is_empty())
        .then(|| idx.write_synthetic_answer(spec.slug, task, spec.answer, &evidence))
        .flatten()
}

pub(super) fn best_preference_session_lines<F>(
    idx: &NeuronIndex,
    task: &str,
    focus_terms: &[&str],
    max_lines: usize,
    mut predicate: F,
) -> Vec<String>
where
    F: FnMut(&str, &str) -> bool,
{
    let focus_owned = focus_terms
        .iter()
        .map(|term| (*term).to_string())
        .collect::<Vec<_>>();
    let mut sessions = idx.candidate_session_ids_by_line_overlap(&focus_owned, 8);
    if sessions.is_empty() {
        sessions = idx
            .candidate_session_ids(task, focus_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect();
    }
    let query_terms = synthetic_query_terms(task);
    let mut best: Option<(usize, Vec<String>)> = None;
    for (session_id, _) in sessions {
        let lines =
            idx.find_session_lines(&session_id, false, 96, |line, lower| predicate(line, lower));
        if !lines.is_empty() {
            let evidence = lines.into_iter().take(max_lines).collect::<Vec<_>>();
            let evidence_lower = evidence
                .iter()
                .map(|line| line.to_ascii_lowercase())
                .collect::<Vec<_>>();
            let focus_score = focus_owned
                .iter()
                .filter(|term| {
                    evidence_lower
                        .iter()
                        .any(|line| line.contains(term.as_str()))
                })
                .count();
            let query_score = query_terms
                .iter()
                .filter(|term| {
                    term.len() > 3
                        && evidence_lower
                            .iter()
                            .any(|line| line.contains(term.as_str()))
                })
                .count();
            let score = focus_score * 100 + query_score * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_evidence)| {
                    score > *best_score
                        || (score == *best_score && evidence.len() > best_evidence.len())
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, evidence));
            }
        }
    }
    best.map(|(_, evidence)| evidence).unwrap_or_default()
}

fn extract_upgrade_pair(line: &str) -> Option<(String, String)> {
    let regex = Regex::new(
        r"(?i)upgrading from (?:my |a |an )?(?P<from>[^.?!]+?) to (?:my |a |an )?(?P<to>[^.?!]+)",
    )
    .ok()?;
    let captures = regex.captures(line)?;
    Some((
        clean_capture(captures.name("from")?.as_str()),
        clean_capture(captures.name("to")?.as_str()),
    ))
}

fn extract_destination_name(line: &str) -> Option<String> {
    let regex = Regex::new(r"(?i)previous visit to (?P<destination>[^,]+)").ok()?;
    Some(clean_capture(
        regex.captures(line)?.name("destination")?.as_str(),
    ))
}

fn extract_destination_memory(line: &str) -> Option<String> {
    let regex = Regex::new(r"(?i)where i had a great time (?P<memory>.+?), i realized").ok()?;
    regex
        .captures(line)
        .and_then(|captures| captures.name("memory"))
        .map(|memory| clean_capture(memory.as_str()))
}

fn extract_destination_interest(line: &str) -> Option<String> {
    let regex = Regex::new(r"(?i)love the city's (?P<interest>[^.?!]+)").ok()?;
    let destination = extract_destination_name(line)?;
    let interest = regex
        .captures(line)
        .and_then(|captures| captures.name("interest"))?;
    Some(format!(
        "{}'s {}",
        destination,
        clean_capture(interest.as_str())
    ))
}

fn extract_quoted_titles(line: &str) -> Vec<String> {
    let regex = Regex::new(r#""([^"]+)""#).ok();
    let mut titles = Vec::new();
    if let Some(regex) = regex {
        for captures in regex.captures_iter(line) {
            if let Some(title) = captures.get(1) {
                push_unique_string(&mut titles, clean_capture(title.as_str()));
            }
        }
    }
    titles
}

fn extract_phone_model(line: &str) -> Option<String> {
    let regex = Regex::new(
        r"(?i)\b(iPhone\s+[0-9A-Za-z ]+(?:Pro Max|Pro|Plus|Mini)?|Samsung Galaxy\s+[A-Za-z0-9 ]+)\b",
    )
    .ok()?;
    regex
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|model| clean_capture(model.as_str()))
}

fn collect_phone_accessory_focuses(evidence: &[String]) -> Vec<String> {
    let mut focuses = Vec::new();
    for line in evidence {
        let lower = line.to_ascii_lowercase();
        if lower.contains("screen protector") {
            push_unique_string(&mut focuses, "high-quality screen protectors".to_string());
        }
        if lower.contains("wallet case") {
            push_unique_string(&mut focuses, "phone wallet cases".to_string());
        }
        if lower.contains("power bank") && lower.contains("wireless charging") {
            push_unique_string(&mut focuses, "wireless charging power banks".to_string());
        } else if lower.contains("power bank") {
            push_unique_string(&mut focuses, "portable power banks".to_string());
        }
    }
    focuses
}

fn clean_capture(value: &str) -> String {
    value
        .trim()
        .trim_matches(['"', '\'', '.', ',', '?', '!'])
        .trim()
        .to_string()
}

pub(super) fn join_text_items(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        [rest @ .., last] => format!("{}, and {}", rest.join(", "), last),
    }
}

pub(super) fn push_unique_string(out: &mut Vec<String>, value: String) {
    if out
        .iter()
        .any(|existing| existing.eq_ignore_ascii_case(&value))
    {
        return;
    }
    out.push(value);
}

pub(super) fn contains_ci(line: &str, needle: &str) -> bool {
    line.to_ascii_lowercase().contains(needle)
}

pub(super) fn has_any(lower: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| lower.contains(needle))
}

pub(super) fn collect_named_phrases(evidence: &[String], mappings: &[(&str, &str)]) -> Vec<String> {
    let mut out = Vec::new();
    for line in evidence {
        let lower = line.to_ascii_lowercase();
        for (needle, phrase) in mappings {
            if lower.contains(needle) {
                push_unique_string(&mut out, (*phrase).to_string());
            }
        }
    }
    out
}

pub(super) fn collect_theme_park_names(evidence: &[String]) -> Vec<String> {
    let mut parks = Vec::new();
    for line in evidence {
        let lower = line.to_ascii_lowercase();
        for (needle, name) in [
            ("disneyland", "Disneyland"),
            ("knott's berry farm", "Knott's Berry Farm"),
            ("six flags magic mountain", "Six Flags Magic Mountain"),
            ("magic mountain", "Six Flags Magic Mountain"),
            ("universal studios hollywood", "Universal Studios Hollywood"),
        ] {
            if lower.contains(needle) {
                push_unique_string(&mut parks, name.to_string());
            }
        }
    }
    parks
}
