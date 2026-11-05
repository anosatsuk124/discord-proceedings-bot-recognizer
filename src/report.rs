//! 議事録を人が読む形にする。1 ファイルで完結する HTML と、Markdown。
//!
//! どちらも外部のファイルやスクリプトを読み込まない。発言者の表示名は Discord の参加者が
//! 自由に決められるので、HTML でも Markdown でも、書式として解釈されないようにエスケープする。

use std::fmt::{self, Write as _};

use jiff::Zoned;
use jiff::tz::TimeZone;

use crate::proceedings::{Proceedings, Row};

/// 発言者を色分けする色の数。HTML の CSS(.s0〜.s7)と合わせる。
const SPEAKER_COLORS: usize = 8;

/// 議事録を Markdown にする。
pub fn render_markdown(proceedings: &Proceedings, timezone: &TimeZone) -> String {
    let mut out = String::with_capacity(1024 + proceedings.rows.len() * 128);
    write_markdown(&mut out, proceedings, timezone).expect("writing to a String cannot fail");
    out
}

/// 議事録を、1 ファイルで完結する HTML にする。
pub fn render_html(proceedings: &Proceedings, timezone: &TimeZone) -> String {
    let mut out = String::with_capacity(STYLE.len() + 2048 + proceedings.rows.len() * 256);
    write_html(&mut out, proceedings, timezone).expect("writing to a String cannot fail");
    out
}

fn write_markdown(out: &mut String, p: &Proceedings, timezone: &TimeZone) -> fmt::Result {
    let started = p.started_at.to_zoned(timezone.clone());
    writeln!(out, "# {}", markdown_text(&title(p, &started)))?;
    writeln!(out)?;
    if let Some(guild) = &p.guild_name {
        writeln!(out, "- サーバー: {}", markdown_text(guild))?;
    }
    if let Some(channel) = &p.channel_name {
        writeln!(out, "- チャンネル: #{}", markdown_text(channel))?;
    }
    writeln!(out, "- 日時: {}", period(p, timezone))?;
    writeln!(out, "- 長さ: {}", length(p))?;
    let speakers = speakers(&p.rows);
    if !speakers.is_empty() {
        let list: Vec<String> = speakers
            .iter()
            .map(|(name, count)| format!("{}({count} 件)", markdown_text(name)))
            .collect();
        writeln!(out, "- 参加者: {}", list.join("、"))?;
    }
    writeln!(out, "- 発言: {} 件", p.rows.len())?;
    writeln!(out)?;
    writeln!(out, "## 発言")?;
    writeln!(out)?;
    if p.rows.is_empty() {
        writeln!(out, "発言はありませんでした。")?;
        return Ok(());
    }
    let mut current_date = started.date();
    for row in &p.rows {
        let at = row.date.to_zoned(timezone.clone());
        // 日付をまたいだら見出しを入れる(行には時刻しか書かないので)。
        if at.date() != current_date {
            current_date = at.date();
            writeln!(out, "\n### {current_date}\n")?;
        }
        writeln!(
            out,
            "- **{} {}**: {}",
            at.strftime("%H:%M:%S"),
            markdown_text(&row.speaker),
            markdown_text(&row.talk)
        )?;
    }
    Ok(())
}

fn write_html(out: &mut String, p: &Proceedings, timezone: &TimeZone) -> fmt::Result {
    let started = p.started_at.to_zoned(timezone.clone());
    let title = title(p, &started);
    let speakers = speakers(&p.rows);

    writeln!(
        out,
        "<!doctype html>\n<html lang=\"ja\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <meta name=\"color-scheme\" content=\"light dark\">\n\
         <title>{}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n<div class=\"page\">",
        html_text(&title)
    )?;
    write_html_header(out, p, timezone, &title, &speakers)?;
    writeln!(out, "<main>")?;
    write_html_lines(out, p, timezone, &speakers)?;
    writeln!(out, "</main>")?;
    writeln!(
        out,
        "<footer>Discord の録音から自動で文字起こしした議事録です。聞き取りの誤りを含むことがあります。<br>セッション {}</footer>",
        html_text(&p.session_id)
    )?;
    writeln!(out, "</div>\n</body>\n</html>")
}

/// 見出し、日時などの情報、参加者の一覧。
fn write_html_header(
    out: &mut String,
    p: &Proceedings,
    timezone: &TimeZone,
    title: &str,
    speakers: &[(&str, usize)],
) -> fmt::Result {
    writeln!(out, "<header>")?;
    writeln!(out, "<p class=\"eyebrow\">議事録</p>")?;
    match &p.channel_name {
        Some(channel) => writeln!(out, "<h1>#{}</h1>", html_text(channel))?,
        None => writeln!(out, "<h1>{}</h1>", html_text(title))?,
    }
    writeln!(out, "<dl class=\"meta\">")?;
    if let Some(guild) = &p.guild_name {
        writeln!(out, "<dt>サーバー</dt><dd>{}</dd>", html_text(guild))?;
    }
    writeln!(
        out,
        "<dt>日時</dt><dd>{}</dd>",
        html_text(&period(p, timezone))
    )?;
    writeln!(out, "<dt>長さ</dt><dd>{}</dd>", length(p))?;
    writeln!(out, "<dt>発言</dt><dd>{} 件</dd>", p.rows.len())?;
    writeln!(out, "</dl>")?;
    if !speakers.is_empty() {
        writeln!(out, "<ul class=\"speakers\" aria-label=\"参加者\">")?;
        for (index, (name, count)) in speakers.iter().enumerate() {
            writeln!(
                out,
                "<li class=\"s{}\">{} <span class=\"count\">{count} 件</span></li>",
                index % SPEAKER_COLORS,
                html_text(name)
            )?;
        }
        writeln!(out, "</ul>")?;
    }
    writeln!(out, "</header>")
}

/// 発言の行。同じ人が続けて話した行は 1 つのまとまり(section)にし、日付をまたいだら区切りを入れる。
fn write_html_lines(
    out: &mut String,
    p: &Proceedings,
    timezone: &TimeZone,
    speakers: &[(&str, usize)],
) -> fmt::Result {
    if p.rows.is_empty() {
        return writeln!(out, "<p class=\"empty\">発言はありませんでした。</p>");
    }
    let mut current_date = p.started_at.to_zoned(timezone.clone()).date();
    let mut current_speaker: Option<&str> = None;
    for (index, row) in p.rows.iter().enumerate() {
        let at = row.date.to_zoned(timezone.clone());
        let new_date = at.date() != current_date;
        if new_date || current_speaker != Some(row.speaker.as_str()) {
            if current_speaker.is_some() {
                writeln!(out, "</section>")?;
            }
            if new_date {
                current_date = at.date();
                writeln!(
                    out,
                    "<h2 class=\"date\"><time datetime=\"{current_date}\">{current_date}</time></h2>"
                )?;
            }
            writeln!(
                out,
                "<section class=\"turn s{}\">\n<p class=\"who\">{}</p>",
                speaker_color(speakers, &row.speaker),
                html_text(&row.speaker)
            )?;
            current_speaker = Some(&row.speaker);
        }
        let id = index + 1;
        writeln!(
            out,
            "<p class=\"line\" id=\"l{id}\"><a class=\"time\" href=\"#l{id}\"><time datetime=\"{}\">{}</time></a><span class=\"talk\">{}</span></p>",
            at.strftime("%Y-%m-%dT%H:%M:%S%:z"),
            at.strftime("%H:%M:%S"),
            html_text(&row.talk)
        )?;
    }
    writeln!(out, "</section>")
}

/// 発言者の色の番号(CSS の .s0〜.s7)。初めて話した順に割り当てる。
fn speaker_color(speakers: &[(&str, usize)], name: &str) -> usize {
    speakers
        .iter()
        .position(|(speaker, _)| *speaker == name)
        .unwrap_or(0)
        % SPEAKER_COLORS
}

/// 例: `議事録 #定例 2026-11-05 23:49`。チャンネル名が無ければ `議事録 2026-11-05 23:49`。
fn title(p: &Proceedings, started: &Zoned) -> String {
    let when = started.strftime("%Y-%m-%d %H:%M");
    match &p.channel_name {
        Some(channel) => format!("議事録 #{channel} {when}"),
        None => format!("議事録 {when}"),
    }
}

/// 例: `2026-11-05 23:49 〜 23:58(JST)`。日付をまたいだら終わりにも日付を付ける。
fn period(p: &Proceedings, timezone: &TimeZone) -> String {
    let started = p.started_at.to_zoned(timezone.clone());
    let ended = p.ended_at.to_zoned(timezone.clone());
    let end_format = if ended.date() == started.date() {
        "%H:%M"
    } else {
        "%Y-%m-%d %H:%M"
    };
    format!(
        "{} 〜 {}({})",
        started.strftime("%Y-%m-%d %H:%M"),
        ended.strftime(end_format),
        started.strftime("%Z")
    )
}

/// 録音の長さ。例: `29 秒`、`5 分 3 秒`、`1 時間 2 分`。
fn length(p: &Proceedings) -> String {
    let seconds = (p.ended_at.as_second() - p.started_at.as_second()).max(0);
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if hours > 0 {
        format!("{hours} 時間 {minutes} 分")
    } else if minutes > 0 {
        format!("{minutes} 分 {seconds} 秒")
    } else {
        format!("{seconds} 秒")
    }
}

/// 初めて話した順の発言者と、その発言数。
fn speakers(rows: &[Row]) -> Vec<(&str, usize)> {
    let mut speakers: Vec<(&str, usize)> = Vec::new();
    for row in rows {
        match speakers.iter_mut().find(|(name, _)| *name == row.speaker) {
            Some((_, count)) => *count += 1,
            None => speakers.push((&row.speaker, 1)),
        }
    }
    speakers
}

fn html_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Markdown の書式やリンク、HTML として解釈される記号をバックスラッシュでエスケープする。
/// 行頭に来ることはない(必ず `- **時刻 名前**: ` などの後ろに書く)ので、行頭でだけ
/// 意味を持つ `#` `-` `+` `.` はそのままにする。改行は空白にして 1 行に収める。
fn markdown_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '|' | '~' | '&' | '$' => {
                out.push('\\');
                out.push(c);
            }
            '\n' | '\r' | '\t' => out.push(' '),
            _ => out.push(c),
        }
    }
    out
}

const STYLE: &str = r#"
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --mark: #f1f4f8;
  --s0: #0969da; --s1: #bf3989; --s2: #1a7f37; --s3: #9a6700;
  --s4: #8250df; --s5: #0e7a86; --s6: #cf222e; --s7: #6e7781;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --mark: #1c2129;
    --s0: #58a6ff; --s1: #f778ba; --s2: #3fb950; --s3: #d29922;
    --s4: #bc8cff; --s5: #39c5cf; --s6: #ff7b72; --s7: #9198a1;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: system-ui, -apple-system, "Hiragino Sans", "Hiragino Kaku Gothic ProN",
    "Noto Sans JP", "Yu Gothic UI", Meiryo, sans-serif;
  font-size: 16px; line-height: 1.8; -webkit-text-size-adjust: 100%;
}
.page { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1rem 4rem; }
header { border-bottom: 1px solid var(--line); padding-bottom: 1.5rem; margin-bottom: 2rem; }
.eyebrow { margin: 0; color: var(--muted); font-size: .875rem; letter-spacing: .08em; }
h1 { margin: .25rem 0 1rem; font-size: 1.75rem; line-height: 1.35; overflow-wrap: anywhere; }
.meta { display: grid; grid-template-columns: max-content 1fr; gap: .2rem 1.25rem; margin: 0 0 1.25rem; }
.meta dt { color: var(--muted); }
.meta dd { margin: 0; overflow-wrap: anywhere; }
.speakers { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: .5rem; }
.speakers li {
  display: inline-flex; align-items: center; gap: .45rem; padding: .1rem .7rem;
  border: 1px solid var(--line); border-radius: 999px; font-size: .875rem; overflow-wrap: anywhere;
}
.speakers li::before { content: ""; flex: none; width: .6rem; height: .6rem; border-radius: 50%; background: var(--c); }
.count { color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.date {
  display: flex; align-items: center; gap: .75rem; margin: 2rem 0 1rem;
  color: var(--muted); font-size: .875rem; font-weight: 400;
}
.date::after { content: ""; flex: 1; border-top: 1px solid var(--line); }
.turn { border-left: 3px solid var(--c); padding: .1rem 0 .1rem 1rem; margin: 0 0 1.5rem; }
.who { margin: 0 0 .2rem; color: var(--c); font-weight: 700; overflow-wrap: anywhere; }
.line { display: grid; grid-template-columns: 4.5rem 1fr; gap: .75rem; margin: 0; border-radius: 4px; }
.line:target { background: var(--mark); }
.time {
  padding-top: .15em; color: var(--muted); font-size: .8125rem;
  font-variant-numeric: tabular-nums; text-decoration: none;
}
.time:hover { text-decoration: underline; }
.talk { overflow-wrap: anywhere; }
.empty { color: var(--muted); }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .8125rem; }
.s0 { --c: var(--s0); } .s1 { --c: var(--s1); } .s2 { --c: var(--s2); } .s3 { --c: var(--s3); }
.s4 { --c: var(--s4); } .s5 { --c: var(--s5); } .s6 { --c: var(--s6); } .s7 { --c: var(--s7); }
@media (max-width: 30rem) {
  .line { grid-template-columns: 1fr; gap: 0; }
  .time { padding-top: 0; }
}
@media print {
  .page { max-width: none; padding: 0; }
  .line { break-inside: avoid; }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use jiff::Timestamp;

    fn row(speaker: &str, date: &str, talk: &str) -> Row {
        Row {
            speaker: speaker.to_owned(),
            date: date.parse::<Timestamp>().unwrap(),
            talk: talk.to_owned(),
        }
    }

    fn sample() -> Proceedings {
        Proceedings {
            session_id: "20261105T144500Z-4aaa7159".to_owned(),
            guild_name: Some("開発部".to_owned()),
            channel_name: Some("定例".to_owned()),
            started_at: "2026-11-05T14:45:00Z".parse().unwrap(),
            ended_at: "2026-11-05T15:20:30Z".parse().unwrap(),
            rows: vec![
                row("山田", "2026-11-05T14:49:05Z", "始めます。"),
                row("山田", "2026-11-05T14:49:10Z", "議題は 2 つです。"),
                row("佐藤", "2026-11-05T14:59:59Z", "はい。"),
                row("佐藤", "2026-11-05T15:00:01Z", "日付が変わりました。"),
            ],
        }
    }

    fn tokyo() -> TimeZone {
        TimeZone::get("Asia/Tokyo").unwrap()
    }

    #[test]
    fn markdown_lists_every_line_with_its_time() {
        assert_eq!(
            render_markdown(&sample(), &tokyo()),
            "# 議事録 #定例 2026-11-05 23:45\n\
             \n\
             - サーバー: 開発部\n\
             - チャンネル: #定例\n\
             - 日時: 2026-11-05 23:45 〜 2026-11-06 00:20(JST)\n\
             - 長さ: 35 分 30 秒\n\
             - 参加者: 山田(2 件)、佐藤(2 件)\n\
             - 発言: 4 件\n\
             \n\
             ## 発言\n\
             \n\
             - **23:49:05 山田**: 始めます。\n\
             - **23:49:10 山田**: 議題は 2 つです。\n\
             - **23:59:59 佐藤**: はい。\n\
             \n\
             ### 2026-11-06\n\
             \n\
             - **00:00:01 佐藤**: 日付が変わりました。\n"
        );
    }

    #[test]
    fn markdown_escapes_formatting_in_names_and_talk() {
        let mut p = sample();
        p.rows = vec![row(
            "**[偽](https://example.com)**",
            "2026-11-05T14:49:05Z",
            "<script>alert(1)</script> | `x` & $y$\nz",
        )];
        let markdown = render_markdown(&p, &tokyo());
        assert!(markdown.contains(
            "- **23:49:05 \\*\\*\\[偽\\](https://example.com)\\*\\***: \\<script\\>alert(1)\\</script\\> \\| \\`x\\` \\& \\$y\\$ z\n"
        ));
    }

    #[test]
    fn html_groups_consecutive_lines_of_the_same_speaker() {
        let html = render_html(&sample(), &tokyo());
        assert!(html.starts_with("<!doctype html>\n<html lang=\"ja\">"));
        assert!(html.contains("<title>議事録 #定例 2026-11-05 23:45</title>"));
        assert!(html.contains("<h1>#定例</h1>"));
        // 山田の 2 行で 1 つ、佐藤は日付をまたぐので 2 つ。
        assert_eq!(html.matches("<section class=\"turn").count(), 3);
        assert_eq!(
            html.matches("<section").count(),
            html.matches("</section>").count()
        );
        assert!(html.contains("<section class=\"turn s0\">\n<p class=\"who\">山田</p>"));
        assert!(html.contains("<section class=\"turn s1\">\n<p class=\"who\">佐藤</p>"));
        assert!(
            html.contains(
                "<h2 class=\"date\"><time datetime=\"2026-11-06\">2026-11-06</time></h2>"
            )
        );
        assert!(html.contains(
            "<p class=\"line\" id=\"l4\"><a class=\"time\" href=\"#l4\"><time datetime=\"2026-11-06T00:00:01+09:00\">00:00:01</time></a><span class=\"talk\">日付が変わりました。</span></p>"
        ));
        // 外部のファイルやスクリプトを読み込まない。
        assert!(!html.contains("<script"));
        assert!(!html.contains("<link"));
        assert!(!html.contains("http"));
    }

    #[test]
    fn html_escapes_names_and_talk() {
        let mut p = sample();
        p.channel_name = Some("<b>ch</b>".to_owned());
        p.rows = vec![row(
            "\"><img src=x onerror=alert(1)>",
            "2026-11-05T14:49:05Z",
            "<script>alert('x')</script> & more",
        )];
        let html = render_html(&p, &tokyo());
        assert!(!html.contains("<img"));
        assert!(!html.contains("<script"));
        assert!(!html.contains("<b>ch"));
        assert!(html.contains("&quot;&gt;&lt;img src=x onerror=alert(1)&gt;"));
        assert!(html.contains("&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt; &amp; more"));
    }

    #[test]
    fn says_so_when_nobody_spoke() {
        let mut p = sample();
        p.rows.clear();
        assert!(render_markdown(&p, &tokyo()).ends_with("## 発言\n\n発言はありませんでした。\n"));
        let html = render_html(&p, &tokyo());
        assert!(html.contains("<p class=\"empty\">発言はありませんでした。</p>"));
        assert!(!html.contains("<section"));
    }

    #[test]
    fn titles_and_lengths_without_channel_names() {
        let mut p = sample();
        p.guild_name = None;
        p.channel_name = None;
        p.ended_at = "2026-11-05T16:47:00Z".parse().unwrap();
        let markdown = render_markdown(&p, &TimeZone::UTC);
        assert!(markdown.starts_with("# 議事録 2026-11-05 14:45\n\n- 日時: 2026-11-05 14:45 〜 16:47(UTC)\n- 長さ: 2 時間 2 分\n"));
        p.ended_at = "2026-11-05T14:45:29Z".parse().unwrap();
        assert!(render_markdown(&p, &TimeZone::UTC).contains("- 長さ: 29 秒\n"));
    }
}
