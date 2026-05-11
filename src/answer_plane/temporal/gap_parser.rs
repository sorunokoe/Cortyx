//! Gap/elapsed query parsing: temporal gap detection and duration event parsing.

use super::*;

pub(crate) fn parse_temporal_gap_query(task: &str) -> Option<TemporalGapQuery> {
    if task.to_ascii_lowercase().starts_with("how many days") {
        if let Some((start, end)) = parse_temporal_duration_events(task) {
            return Some(TemporalGapQuery {
                start,
                end: TemporalGapEndpoint::Event(end),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: "day".to_string(),
                },
            });
        }
    }
    parse_temporal_explicit_unit_gap_query(task)
        .or_else(|| parse_temporal_how_long_gap_query(task))
        .or_else(|| {
            let (start, end) = parse_temporal_duration_events(task)?;
            Some(TemporalGapQuery {
                start,
                end: TemporalGapEndpoint::Event(end),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: "day".to_string(),
                },
            })
        })
}

fn parse_temporal_explicit_unit_gap_query(task: &str) -> Option<TemporalGapQuery> {
    let trimmed = task.trim().trim_end_matches('?');
    let lower = trimmed.to_ascii_lowercase();
    for unit in ["day", "week", "month", "year"] {
        let prefixes = [
            format!("how many {unit} "),
            format!("how many {unit}s "),
            format!("how many {unit}"),
            format!("how many {unit}s"),
        ];
        if !prefixes
            .iter()
            .any(|prefix| lower.starts_with(prefix.trim_end()))
        {
            continue;
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s had passed between "))
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} had passed between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit}s passed between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} passed between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit}s were there between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} were there between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit}s passed between the time "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} passed between the time "),
                    )
                })
        {
            let (left, right) = split_once_case_insensitive(rest, " and ")?;
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(left)?,
                end: TemporalGapEndpoint::Event(build_temporal_event_option(right)?),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s before ")).or_else(
                || strip_prefix_case_insensitive(trimmed, &format!("How many {unit} before ")),
            )
        {
            let (reference, target) = split_once_case_insensitive(rest, " did ")?;
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(target)?,
                end: TemporalGapEndpoint::Event(build_temporal_event_option(reference)?),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s after ")).or_else(
                || strip_prefix_case_insensitive(trimmed, &format!("How many {unit} after ")),
            )
        {
            let (reference, target) = split_once_case_insensitive(rest, " did ")?;
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(reference)?,
                end: TemporalGapEndpoint::Event(build_temporal_event_option(target)?),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }

        let take_markers = ["did it take for ".to_string(), "did it take me to ".to_string()];
        for marker in take_markers {
            if let Some(idx) = lower.find(&marker) {
                let rest = &trimmed[idx + marker.len()..];
                let (target, start) = split_once_case_insensitive(rest, " after ")?;
                return Some(TemporalGapQuery {
                    start: build_temporal_event_option(start)?,
                    end: TemporalGapEndpoint::Event(build_temporal_event_option(target)?),
                    answer_style: TemporalGapAnswerStyle::FixedUnit {
                        unit: unit.to_string(),
                    },
                });
            }
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s had passed since "))
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} had passed since "),
                    )
                })
        {
            if let Some((start, end)) = split_once_case_insensitive(rest, " when ") {
                return Some(TemporalGapQuery {
                    start: build_temporal_event_option(start)?,
                    end: TemporalGapEndpoint::Event(build_temporal_event_option(end)?),
                    answer_style: TemporalGapAnswerStyle::FixedUnit {
                        unit: unit.to_string(),
                    },
                });
            }
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s have passed since "))
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} have passed since "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit}s has passed since "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} has passed since "),
                    )
                })
        {
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(rest)?,
                end: TemporalGapEndpoint::CurrentMoment,
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s have I been "))
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit} have I been "))
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s had I been "))
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit} had I been "))
                })
        {
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(rest)?,
                end: TemporalGapEndpoint::CurrentMoment,
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }
    }
    None
}

fn parse_temporal_how_long_gap_query(task: &str) -> Option<TemporalGapQuery> {
    let trimmed = task.trim().trim_end_matches('?');
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("how long") && !lower.starts_with("how much time") {
        return None;
    }

    if let Some((_left, right)) = split_once_case_insensitive(trimmed, " between ")
        .or_else(|| split_once_case_insensitive(trimmed, " did "))
    {
        if let Some((start, end)) = split_once_case_insensitive(right, " and ") {
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(start)?,
                end: TemporalGapEndpoint::Event(build_temporal_event_option(end)?),
                answer_style: TemporalGapAnswerStyle::NaturalLanguage,
            });
        }
    }
    None
}

pub(crate) fn parse_temporal_duration_events(task: &str) -> Option<(ChoiceOption, ChoiceOption)> {
    let trimmed = task.trim().trim_end_matches('?');
    let lower = trimmed.to_ascii_lowercase();
    if !lower.contains("how many days") {
        return None;
    }

    if let Some(rest) = trimmed
        .strip_prefix("How many days had passed between ")
        .or_else(|| trimmed.strip_prefix("How many days passed between "))
        .or_else(|| trimmed.strip_prefix("How many days were there between "))
    {
        let (left, right) = split_once_case_insensitive(rest, " and ")?;
        return Some((
            build_temporal_event_option(left)?,
            build_temporal_event_option(right)?,
        ));
    }

    if let Some(rest) = trimmed.strip_prefix("How many days before ") {
        let (reference, target) = split_once_case_insensitive(rest, " did ")?;
        return Some((
            build_temporal_event_option(target)?,
            build_temporal_event_option(reference)?,
        ));
    }

    if let Some(rest) = trimmed.strip_prefix("How many days after ") {
        let (reference, target) = split_once_case_insensitive(rest, " did ")?;
        return Some((
            build_temporal_event_option(reference)?,
            build_temporal_event_option(target)?,
        ));
    }

    let take_marker = "did it take for ";
    if let Some(idx) = lower.find(take_marker) {
        let rest = &trimmed[idx + take_marker.len()..];
        let (target, start) = split_once_case_insensitive(rest, " after ")?;
        return Some((
            build_temporal_event_option(start)?,
            build_temporal_event_option(target)?,
        ));
    }

    None
}

pub(crate) fn build_temporal_event_option(text: &str) -> Option<ChoiceOption> {
    let display = strip_leading_temporal_actor(text);
    let mut tokens = display
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter_map(|token| {
            let lower = token
                .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                .trim_matches('\'')
                .to_ascii_lowercase();
            if lower.len() < 3
                || (QUESTION_STOPWORDS.contains(&lower.as_str())
                    && !matches!(lower.as_str(), "book" | "booked" | "booking"))
                || parse_count_token(&lower).is_some()
            {
                None
            } else {
                Some(lower)
            }
        })
        .collect::<Vec<_>>();
    if tokens.iter().filter(|token| token.len() >= 4).count() >= 2 {
        tokens.retain(|token| token.len() >= 4);
    }
    tokens.sort();
    tokens.dedup();
    if display.is_empty() || tokens.is_empty() {
        return None;
    }
    Some(ChoiceOption { display, tokens })
}

pub(crate) fn strip_leading_temporal_actor(text: &str) -> String {
    let mut clean = sanitize_answer_text(text);
    loop {
        let mut stripped = false;
        for prefix in [
            "the day i ",
            "the time i ",
            "the day ",
            "the time ",
            "day i ",
            "time i ",
            "when i ",
            "i ",
            "me ",
            "my ",
            "we ",
            "our ",
            "he ",
            "his ",
            "she ",
            "her ",
            "they ",
            "their ",
            "to ",
        ] {
            if clean.to_ascii_lowercase().starts_with(prefix) {
                clean = clean[prefix.len()..].trim().to_string();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    clean
}

pub(crate) fn split_once_case_insensitive<'a>(
    text: &'a str,
    delimiter: &str,
) -> Option<(&'a str, &'a str)> {
    strip_prefix_case_insensitive(text, "").and_then(|_| {
        let lower = text.to_ascii_lowercase();
        let lower_delim = delimiter.to_ascii_lowercase();
        lower
            .find(&lower_delim)
            .map(|idx| (&text[..idx], &text[idx + delimiter.len()..]))
    })
}

fn strip_prefix_case_insensitive<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let lower_text = text.to_ascii_lowercase();
    let lower_prefix = prefix.to_ascii_lowercase();
    if lower_text.starts_with(&lower_prefix) {
        Some(&text[prefix.len()..])
    } else {
        None
    }
}

pub(crate) fn parse_temporal_elapsed_query(task: &str) -> Option<(String, ChoiceOption)> {
    let trimmed = task.trim().trim_end_matches('?');
    if !trimmed.to_ascii_lowercase().starts_with("how many ") {
        return None;
    }
    for unit in ["day", "week", "month", "year"] {
        for marker in [format!("{unit} ago did "), format!("{unit}s ago did ")] {
            let Some((_, event)) = split_once_case_insensitive(trimmed, &marker) else {
                continue;
            };
            return Some((unit.to_string(), build_temporal_event_option(event)?));
        }
    }
    None
}

pub(crate) fn parse_temporal_order_direction(task: &str) -> Option<TemporalDirection> {
    let lower = task.to_ascii_lowercase();
    if lower.contains(" first")
        || lower.starts_with("first ")
        || lower.contains(" earliest")
        || lower.contains(" older")
    {
        Some(TemporalDirection::Earlier)
    } else if lower.contains(" last")
        || lower.contains(" latest")
        || lower.contains(" most recent")
        || lower.contains(" newest")
    {
        Some(TemporalDirection::Later)
    } else {
        None
    }
}
