//! Frozen RED tests for T-08.02 — add the parity health endpoint.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must build in `syrinx-serve`:
//!
//!   * `router() -> axum::Router` — the existing OpenAI-compatible audio router,
//!     now also exposing a `GET /v1/health` route registered exactly once.
//!   * `Health { status, version }` — the documented typed JSON health body. It
//!     derives `serde::Deserialize` so a `/v1/health` response body deserializes
//!     straight into it. `status` is the literal health marker `"ok"`; `version`
//!     is a non-empty version string.
//!
//! Contract (list.md / DESIGN §T8.2): a `GET` to `/v1/health` returns 200 with an
//! `application/json` body that deserializes into the typed `Health` struct whose
//! `status` is `"ok"` and whose `version` is non-empty; a non-`GET` method to the
//! same path returns 405 Method Not Allowed (the path exists for exactly the one
//! method); and the route is registered exactly once (building the router does not
//! panic on a duplicate, the path answers `GET` with 200 — not 404 — and an
//! unrelated path is 404).
//!
//! Out of scope (not_doing): no liveness/readiness orchestration or dependency
//! probing, and no metrics/telemetry endpoint — just the single typed health body.
//!
//! RED: `syrinx-serve` exposes no `/v1/health` route and no `Health` type yet, so
//! the `Health` symbol does not resolve (the target fails to build) and the route
//! assertions would fail — every criterion is unmet. GREEN adds the route and the
//! typed body so each assertion below holds.

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, Response, StatusCode};
use tower::ServiceExt; // brings `oneshot` into scope

use syrinx_serve::{router, Health};

/// The audited health route — registered exactly once for `GET`.
const HEALTH_ROUTE: &str = "/v1/health";

/// Drive a router with one HTTP request and return the response.
async fn send(app: axum::Router, method: &str, path: &str) -> Response<Body> {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .expect("request must build");
    app.oneshot(request)
        .await
        .expect("the router must answer the request")
}

/// Read the response's content-type header as a string.
fn content_type(response: &Response<Body>) -> String {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .expect("a response must carry a content-type header")
        .to_str()
        .expect("the content-type header must be valid UTF-8")
        .to_string()
}

/// Collect a response body fully into bytes.
async fn collect(response: Response<Body>) -> Vec<u8> {
    to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the response body must collect")
        .to_vec()
}

// ----------------------------------------------------------------------------
// C1 — `/v1/health` returns 200 with the documented typed JSON body.
// ----------------------------------------------------------------------------

/// A `GET` to `/v1/health` returns 200 with an `application/json` content-type and
/// a body that is the documented typed health shape.
#[tokio::test]
async fn test_health_returns_200_typed_json_body() {
    let response = send(router(), "GET", HEALTH_ROUTE).await;

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a GET to the health route must return 200"
    );
    assert_eq!(
        content_type(&response),
        "application/json",
        "the health body must be typed JSON"
    );

    let body = collect(response).await;
    let health: Health =
        serde_json::from_slice(&body).expect("the 200 body must deserialize into the typed Health");
    assert_eq!(
        health.status, "ok",
        "the documented health body must report status \"ok\""
    );
    assert!(
        !health.version.is_empty(),
        "the documented health body must carry a non-empty version"
    );
}

// ----------------------------------------------------------------------------
// C2 — GET -> 200 documented shape; a non-GET method -> 405.
// ----------------------------------------------------------------------------

/// `GET /v1/health` succeeds with 200 and the documented JSON shape (status `"ok"`,
/// non-empty version) — the allowed-method side of the method gate.
#[tokio::test]
async fn test_health_get_returns_200_documented_shape() {
    let response = send(router(), "GET", HEALTH_ROUTE).await;

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "GET is the allowed method and must be 200"
    );

    let body = collect(response).await;
    let health: Health =
        serde_json::from_slice(&body).expect("the GET body must deserialize into the typed Health");
    assert_eq!(health.status, "ok", "GET must return the documented status");
    assert!(
        !health.version.is_empty(),
        "GET must return the documented non-empty version"
    );
}

/// A non-`GET` method to `/v1/health` returns 405 Method Not Allowed — the path is
/// registered for exactly the one method, so other methods are rejected (not 200,
/// not 404). Both `POST` and `DELETE` are non-GET and must be rejected.
#[tokio::test]
async fn test_health_non_get_method_returns_405() {
    let post = send(router(), "POST", HEALTH_ROUTE).await;
    assert_eq!(
        post.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "a POST to the GET-only health route must be 405"
    );

    let delete = send(router(), "DELETE", HEALTH_ROUTE).await;
    assert_eq!(
        delete.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "a DELETE to the GET-only health route must be 405"
    );
}

// ----------------------------------------------------------------------------
// C3 — the route is registered exactly once and the body deserializes into the
//      documented typed health struct.
// ----------------------------------------------------------------------------

/// The health route is registered exactly once: `router()` builds without panicking
/// (a duplicate registration would panic), the path answers `GET` with 200 (not
/// 404), the same path rejects a non-GET method with 405 (it exists for exactly
/// that one method), and an unrelated nearby path is 404.
#[tokio::test]
async fn test_health_route_registered_exactly_once() {
    // Building the router must not panic — a duplicate route registration would.
    let get = send(router(), "GET", HEALTH_ROUTE).await;
    assert_eq!(
        get.status(),
        StatusCode::OK,
        "the health route must be registered and answer GET"
    );

    // The path exists for exactly one method: a non-GET to it is 405, not 404.
    let post = send(router(), "POST", HEALTH_ROUTE).await;
    assert_eq!(
        post.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "the registered path must reject non-GET methods with 405, not 404"
    );

    // An unrelated path beneath the health route is not registered at all -> 404.
    let missing = send(router(), "GET", "/v1/health/extra").await;
    assert_eq!(
        missing.status(),
        StatusCode::NOT_FOUND,
        "an unregistered path must be 404, proving the health route is the only one"
    );
}

/// The returned JSON deserializes into the documented typed `Health` struct: the
/// body parses into `Health` and both documented fields carry their documented
/// values (`status == "ok"`, a non-empty `version`).
#[tokio::test]
async fn test_health_body_deserializes_into_typed_struct() {
    let response = send(router(), "GET", HEALTH_ROUTE).await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = collect(response).await;
    let health: Health = serde_json::from_slice(&body)
        .expect("the health body must deserialize into the documented typed Health struct");

    assert_eq!(
        health.status, "ok",
        "the typed struct's status field must hold the documented value"
    );
    assert!(
        !health.version.is_empty(),
        "the typed struct's version field must be present and non-empty"
    );
}
