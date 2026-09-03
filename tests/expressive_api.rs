//! C3.3. **AC:** an OpenAI-compatible request with inline cues works unchanged;
//! `?explain=1` returns the full report; the caps endpoint is documented in OpenAPI.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::Value;
use tower::ServiceExt;

async fn call(uri: &str, body: Option<&str>) -> (StatusCode, Vec<u8>, String) {
    let app = syrinx_serve::router();
    let req = match body {
        Some(b) => Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(b.to_string()))
            .unwrap(),
        None => Request::builder().uri(uri).body(Body::empty()).unwrap(),
    };
    let res = app.oneshot(req).await.unwrap();
    let status = res.status();
    let ct = res
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_string())
        .unwrap_or_default();
    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap().to_vec();
    (status, bytes, ct)
}

fn speech_body(input: &str) -> String {
    serde_json::json!({ "model": "syrinx", "input": input, "voice": "default" }).to_string()
}

// ------------------------------------------------------------------ unchanged contract

#[tokio::test]
async fn an_openai_request_with_inline_cues_still_returns_audio() {
    // The AC's "works unchanged": cues in `input` must not alter the response contract.
    for input in [
        "hello there",
        "[happy] hello there",
        "[happy] hi [sad] bye",
        "he paused [laughs] and went on",
        "<|speaker:0|> one <|speaker:1|> two",
        "<prosody rate=\"slow\">slowly now</prosody>",
    ] {
        let (status, body, ct) = call("/v1/audio/speech", Some(&speech_body(input))).await;
        assert_eq!(status, StatusCode::OK, "input {input:?} changed the status");
        assert!(!body.is_empty(), "input {input:?} produced no audio");
        assert!(ct.starts_with("audio/"), "input {input:?} changed content-type to {ct}");
    }
}

#[tokio::test]
async fn the_existing_error_contract_is_untouched() {
    let (status, ..) = call("/v1/audio/speech", Some(&speech_body("   "))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "blank input must still be a 400");
    let (status, ..) = call("/v1/audio/speech", Some("{\"model\":\"x\"}")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "malformed body must still be 422");
}

// ------------------------------------------------------------------ ?explain=1

#[tokio::test]
async fn explain_returns_the_full_report_instead_of_audio() {
    let (status, body, ct) = call(
        "/v1/audio/speech?explain=1",
        Some(&speech_body("[happy] hello [wibble] there")),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.starts_with("application/json"), "explain must return JSON, got {ct}");
    let v: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(v["backend"], "fish-s2-pro", "default backend");
    // The spoken text carries no cue markup — the hard invariant, over the wire.
    let text = v["text"].as_str().unwrap();
    assert!(!text.contains('['), "cue markup reached the API response: {text:?}");
    assert!(!text.contains(']'));

    let entries = v["report"]["entries"].as_array().expect("report entries");
    assert_eq!(entries.len(), 2, "one line per cue: {entries:#?}");
    // Every entry carries the author's text, the source range, and an action.
    for e in entries {
        assert!(e["raw"].is_string());
        assert!(e["source"]["start"].is_number());
        assert!(!e["action"].is_null());
    }
    let raws: Vec<_> = entries.iter().map(|e| e["raw"].as_str().unwrap()).collect();
    assert!(raws.contains(&"happy"));
    assert!(raws.contains(&"wibble"));
}

#[tokio::test]
async fn explain_shows_a_cue_being_dropped_and_says_why() {
    // The whole point of the endpoint: a cue that does nothing must be explainable.
    let (status, body, _) = call(
        "/v1/audio/speech?explain=1&backend=qwen3-0.6b-customvoice",
        Some(&speech_body("[happy] hello")),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["backend"], "qwen3-0.6b-customvoice");
    let action = &v["report"]["entries"][0]["action"];
    assert_eq!(
        action["Dropped"]["reason"], "AcceptedButIgnored",
        "the 0.6B silently ignores instructions; the report must say so: {action:#?}"
    );
}

#[tokio::test]
async fn explain_accepts_only_known_backends() {
    let (status, body, _) = call(
        "/v1/audio/speech?explain=1&backend=not-a-backend",
        Some(&speech_body("[happy] hi")),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("/v1/audio/caps"),
            "the error should point at the caps endpoint");
}

#[tokio::test]
async fn explain_is_off_unless_explicitly_requested() {
    // A stray `explain=0` must not silently swallow someone's audio.
    for uri in ["/v1/audio/speech", "/v1/audio/speech?explain=0", "/v1/audio/speech?other=1"] {
        let (status, _, ct) = call(uri, Some(&speech_body("[happy] hi"))).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
        assert!(ct.starts_with("audio/"), "{uri} returned {ct} instead of audio");
    }
}

// ------------------------------------------------------------------ caps endpoint

#[tokio::test]
async fn the_caps_endpoint_returns_every_backend() {
    let (status, body, ct) = call("/v1/audio/caps", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.starts_with("application/json"));
    let v: Value = serde_json::from_slice(&body).unwrap();
    let backends = v["backends"].as_array().unwrap();
    assert_eq!(backends.len(), syrinx_cue::BackendId::ALL.len());
    for b in backends {
        for key in ["id", "krate", "model", "inline", "granularity", "instruct", "source"] {
            assert!(!b[key].is_null(), "caps entry missing {key}: {b:#?}");
        }
    }
    // The accepted/honored distinction must survive to the wire, or clients cannot act on it.
    let small = backends.iter().find(|b| b["id"] == "qwen3-0.6b-customvoice").unwrap();
    let big = backends.iter().find(|b| b["id"] == "qwen3-1.7b-customvoice").unwrap();
    assert_eq!(small["instruct"], "accepted");
    assert_eq!(big["instruct"], "honored");
}

// ------------------------------------------------------------------ OpenAPI

#[test]
fn openapi_documents_the_caps_endpoint_and_the_explain_flag() {
    let spec = std::fs::read_to_string("docs/api/openapi.yaml")
        .expect("docs/api/openapi.yaml must exist");
    for needle in [
        "/v1/audio/caps:",
        "/v1/audio/speech:",
        "CapsResponse",
        "ControlCaps",
        "ExplainResponse",
        "LoweringReport",
        "name: explain",
        "name: backend",
    ] {
        assert!(spec.contains(needle), "openapi.yaml does not document {needle:?}");
    }
    // The Support enum's three levels must be documented, since `accepted` is the one a
    // client is most likely to misread as working.
    assert!(spec.contains("enum: [unsupported, accepted, honored]"));
    assert!(spec.contains("NOT acted on"), "the `accepted` trap must be spelled out");
}

#[test]
fn every_route_the_router_serves_is_documented() {
    let spec = std::fs::read_to_string("docs/api/openapi.yaml").unwrap();
    let src = std::fs::read_to_string("crates/syrinx-serve/src/lib.rs").unwrap();
    // Extract the routes the router actually registers, so a new endpoint cannot ship
    // undocumented.
    let mut routes: Vec<String> = Vec::new();
    for line in src.lines() {
        if let Some(rest) = line.trim().strip_prefix(".route(\"") {
            if let Some(end) = rest.find('"') {
                routes.push(rest[..end].to_string());
            }
        }
    }
    routes.sort();
    routes.dedup();
    assert!(routes.len() >= 4, "route scan found only {routes:?}");
    for r in &routes {
        assert!(spec.contains(&format!("{r}:")), "route {r} is not documented in openapi.yaml");
    }
}
