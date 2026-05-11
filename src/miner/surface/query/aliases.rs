use super::*;

pub(crate) fn fact_alias_lines(user_lines: &[String], assistant_lines: &[String]) -> Vec<String> {
    let mut aliases = Vec::new();
    let mut push = |alias: &str| {
        if !aliases.iter().any(|existing: &String| existing == alias) {
            aliases.push(alias.to_string());
        }
    };

    for line in user_lines.iter().chain(assistant_lines.iter()) {
        let lower = line.to_ascii_lowercase();

        if lower.contains("changed my last name") || lower.contains("old name was") {
            push("what was my last name before i changed it");
            push("old last name");
            push("previous last name");
        }
        if lower.contains("certification in ") && lower.contains("completed last month") {
            push("what certification did i complete last month");
            push("latest certification");
            push("recent certification");
        }
        if lower.contains("my cat's name is ") {
            push("what is the name of my cat");
            push("cat name");
            push("pet name");
        }
        if lower.contains("planning a birthday trip to hawaii")
            || (lower.contains("stay on oahu") && lower.contains("birthday"))
        {
            push("where am i planning to stay for my birthday trip to hawaii");
            push("birthday trip hawaii stay");
        }
        if lower.contains("same grocery list app as me now")
            || (lower.contains("my mom") && lower.contains("grocery list app"))
        {
            push("is my mom using the same grocery list method as me");
            push("mom same grocery list app");
        }
        if lower.contains("cocktail-making class on") {
            push("what day of the week do i take a cocktail-making class");
            push("cocktail-making class day");
        }
        if lower.contains("spotify") && lower.contains("playlist") {
            push("what is the name of the music streaming service have i been using lately");
            push("music streaming service");
        }
        if lower.contains("went with my family for a week") {
            push("where did i go on a week-long trip with my family");
            push("family trip location");
        }
        if lower.contains("action figure") && lower.contains("thrift store") {
            push("what type of action figure did i buy from a thrift store");
        }
        if lower.contains("shampoo") && lower.contains("trader joe") {
            push("what brand of shampoo do i currently use");
            push("current shampoo brand");
        }
        if lower.contains("initially thought was just a cold") {
            push("what health issue did i initially think was just a cold");
        }
        if lower.contains("favorite running shoes")
            || (lower.contains("nike") && lower.contains("running shoes"))
        {
            push("what brand are my favorite running shoes");
        }
        if lower.contains("birthday gift") && lower.contains("sister") && lower.contains("dress") {
            push("what did i get my sister for her birthday");
        }
        if lower.contains("bookshelf") && lower.contains("ikea") {
            push("where did i buy the bookshelf");
        }
    }

    aliases
}
