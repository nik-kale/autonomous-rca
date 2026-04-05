"""Synthetic evidence data for three investigation scenarios.

Every piece of data here is entirely fictional.  The timestamps, hostnames,
pod names, IP addresses, and log messages are invented to provide realistic
signal for the agentic pipeline while including noise entries that exercise
retrieval strategy selection and hypothesis pruning.

Each scenario dictionary has the same shape::

    {
        "problem_statement": str,
        "evidence_sources": {domain: [evidence_items]},
        "expected_root_cause": str,
        "expected_iterations": int,
    }

Evidence items are ``{"timestamp": str, "source": str, "text": str}``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scenario 1 — Kubernetes Disk Pressure
# ---------------------------------------------------------------------------
# Root cause chain: disk usage on worker-3 exceeded threshold → kubelet
# set DiskPressure=True → pod checkout-pod-7b evicted → deployment has
# 0 available replicas → HTTP 503 for customers.

KUBERNETES_DISK_PRESSURE: dict = {
    "problem_statement": (
        "The checkout service (port 8080) is unreachable. "
        "Customers cannot complete orders."
    ),
    "evidence_sources": {
        "app-logs": [
            {
                "timestamp": "2026-04-30T10:02:01Z",
                "source": "checkout-service.log",
                "text": "HTTP 503 — upstream connect error, service 'checkout' unreachable",
            },
            {
                "timestamp": "2026-04-30T10:02:03Z",
                "source": "checkout-service.log",
                "text": "timeout waiting for connection to checkout-pod-7b on port 8080",
            },
            {
                "timestamp": "2026-04-30T10:02:10Z",
                "source": "checkout-service.log",
                "text": "retry 3/3 failed for checkout-pod-7b:8080 — giving up",
            },
            {
                "timestamp": "2026-04-30T10:01:55Z",
                "source": "gateway.log",
                "text": "upstream health check failed for checkout backend pool",
            },
            {
                "timestamp": "2026-04-30T09:50:00Z",
                "source": "checkout-service.log",
                "text": "INFO request processed in 42ms — order #88123 confirmed",
            },
        ],
        "k8s-events": [
            {
                "timestamp": "2026-04-30T10:01:45Z",
                "source": "cluster-state/events.json",
                "text": "Pod checkout-pod-7b evicted from node worker-3",
            },
            {
                "timestamp": "2026-04-30T10:01:44Z",
                "source": "cluster-state/events.json",
                "text": (
                    "Eviction manager: pod checkout-pod-7b exceeds "
                    "ephemeral-storage resource limits"
                ),
            },
            {
                "timestamp": "2026-04-30T10:01:43Z",
                "source": "cluster-state/deployments.json",
                "text": "Deployment 'checkout-app' Available replicas: 0",
            },
            {
                "timestamp": "2026-04-30T10:01:42Z",
                "source": "cluster-state/events.json",
                "text": (
                    "Node worker-3 tainted with "
                    "node.kubernetes.io/disk-pressure:NoSchedule"
                ),
            },
            {
                "timestamp": "2026-04-30T09:45:00Z",
                "source": "cluster-state/events.json",
                "text": "Pod payment-pod-3a running on node worker-2 — healthy",
            },
            {
                "timestamp": "2026-04-30T09:30:00Z",
                "source": "cluster-state/events.json",
                "text": "Scheduled pod checkout-pod-7b on node worker-3",
            },
        ],
        "node-metrics": [
            {
                "timestamp": "2026-04-30T10:00:12Z",
                "source": "worker-3/kubelet.log",
                "text": "kubelet: node worker-3 condition DiskPressure=True",
            },
            {
                "timestamp": "2026-04-30T09:58:30Z",
                "source": "worker-3/kubelet.log",
                "text": (
                    "disk usage on /var/lib/kubelet reached 94%, "
                    "eviction threshold is 90%"
                ),
            },
            {
                "timestamp": "2026-04-30T09:55:00Z",
                "source": "worker-3/syslog",
                "text": (
                    "Container runtime storage: used 32Gi of 34Gi, "
                    "available 2Gi"
                ),
            },
            {
                "timestamp": "2026-04-30T09:50:00Z",
                "source": "worker-3/kubelet.log",
                "text": "Image garbage collection freed 0 bytes — no unused images",
            },
            {
                "timestamp": "2026-04-30T09:45:00Z",
                "source": "worker-1/kubelet.log",
                "text": "kubelet: node worker-1 condition DiskPressure=False",
            },
            {
                "timestamp": "2026-04-30T09:45:00Z",
                "source": "worker-2/kubelet.log",
                "text": "kubelet: node worker-2 condition DiskPressure=False",
            },
            {
                "timestamp": "2026-04-30T09:40:00Z",
                "source": "worker-3/kubelet.log",
                "text": (
                    "disk usage on /var/lib/kubelet reached 88%, "
                    "approaching eviction threshold"
                ),
            },
        ],
        "network-logs": [
            {
                "timestamp": "2026-04-30T10:02:05Z",
                "source": "router-gw1/syslog",
                "text": "GigabitEthernet0/0/0: Interface is up, line protocol is up",
            },
            {
                "timestamp": "2026-04-30T10:01:00Z",
                "source": "router-gw1/syslog",
                "text": "OSPF adjacency on Gi0/0/0 Full, no changes detected",
            },
            {
                "timestamp": "2026-04-30T10:00:00Z",
                "source": "switch-tor1/syslog",
                "text": "No BGP neighbor state changes in last 24 hours",
            },
            {
                "timestamp": "2026-04-30T09:59:00Z",
                "source": "switch-tor1/syslog",
                "text": "All trunk ports operational — VLAN 100,200,300 active",
            },
            {
                "timestamp": "2026-04-30T09:58:00Z",
                "source": "firewall-fw1/syslog",
                "text": "No denied flows to 10.0.0.0/8 in last 60 minutes",
            },
        ],
    },
    "expected_root_cause": (
        "DiskPressure on node worker-3 caused by ephemeral storage "
        "exhaustion (94% usage, 90% threshold). Kubelet evicted "
        "checkout-pod-7b, leaving the checkout-app deployment with "
        "0 available replicas and making the service unreachable."
    ),
    "expected_iterations": 3,
}


# ---------------------------------------------------------------------------
# Scenario 2 — Database Connection Pool Exhaustion
# ---------------------------------------------------------------------------
# Root cause chain: unoptimized analytics query without timeout holds 40
# connections → pool reaches 100/100 → new app requests cannot get a
# connection → response times spike → customers experience timeouts.

DATABASE_CONNECTION_POOL: dict = {
    "problem_statement": (
        "The order-management service is experiencing response times "
        "above 30 seconds. Multiple customers report timeouts when "
        "viewing order history."
    ),
    "evidence_sources": {
        "app-logs": [
            {
                "timestamp": "2026-05-01T14:30:05Z",
                "source": "order-service.log",
                "text": (
                    "WARN connection pool exhausted — waited 28s for "
                    "available connection"
                ),
            },
            {
                "timestamp": "2026-05-01T14:30:12Z",
                "source": "order-service.log",
                "text": (
                    "ERROR request /api/orders/history timed out after 30s"
                ),
            },
            {
                "timestamp": "2026-05-01T14:29:58Z",
                "source": "order-service.log",
                "text": (
                    "WARN HikariPool-1 — pool stats: active=100, idle=0, "
                    "waiting=37, total=100"
                ),
            },
            {
                "timestamp": "2026-05-01T14:25:00Z",
                "source": "order-service.log",
                "text": (
                    "INFO request /api/orders/history completed in 120ms"
                ),
            },
            {
                "timestamp": "2026-05-01T14:31:00Z",
                "source": "order-service.log",
                "text": (
                    "ERROR SqlTransientConnectionException: could not "
                    "acquire connection from pool within 30000ms"
                ),
            },
        ],
        "db-metrics": [
            {
                "timestamp": "2026-05-01T14:30:00Z",
                "source": "postgres-primary/pg_stat_activity",
                "text": (
                    "Active connections: 100/100 max_connections. "
                    "40 connections held by user 'analytics_svc'"
                ),
            },
            {
                "timestamp": "2026-05-01T14:29:45Z",
                "source": "postgres-primary/pg_stat_activity",
                "text": (
                    "Long-running query (pid 8842): SELECT * FROM orders "
                    "JOIN line_items ON orders.id = line_items.order_id "
                    "JOIN products ON line_items.product_id = products.id "
                    "WHERE orders.created_at > '2025-01-01' "
                    "— running for 47 minutes, state: active"
                ),
            },
            {
                "timestamp": "2026-05-01T14:29:50Z",
                "source": "postgres-primary/pg_stat_activity",
                "text": (
                    "analytics_svc connections breakdown: 40 active, "
                    "0 idle. No statement_timeout configured for role."
                ),
            },
            {
                "timestamp": "2026-05-01T14:00:00Z",
                "source": "postgres-primary/pg_stat_activity",
                "text": "Active connections: 58/100 max_connections — healthy",
            },
            {
                "timestamp": "2026-05-01T14:28:00Z",
                "source": "postgres-primary/pg_stat_statements",
                "text": (
                    "Top query by total_time: SELECT * FROM orders "
                    "JOIN line_items ... — 0 index scans, "
                    "sequential scan on orders (12M rows)"
                ),
            },
            {
                "timestamp": "2026-05-01T14:30:10Z",
                "source": "postgres-replica/pg_stat_replication",
                "text": "Replication lag: 0.2s — replica healthy",
            },
        ],
        "infra-metrics": [
            {
                "timestamp": "2026-05-01T14:30:00Z",
                "source": "postgres-primary/node_exporter",
                "text": "CPU usage: 78%, Memory: 62%, Disk I/O wait: 45%",
            },
            {
                "timestamp": "2026-05-01T14:29:00Z",
                "source": "postgres-primary/node_exporter",
                "text": (
                    "Disk read throughput: 450 MB/s (baseline: 50 MB/s) "
                    "— sequential scan detected"
                ),
            },
            {
                "timestamp": "2026-05-01T14:30:00Z",
                "source": "order-service-pod/metrics",
                "text": "Pod CPU: 12%, Memory: 340Mi/512Mi — within limits",
            },
            {
                "timestamp": "2026-05-01T14:30:00Z",
                "source": "load-balancer/metrics",
                "text": (
                    "Backend response time p99: 31200ms (baseline: 180ms)"
                ),
            },
        ],
        "network-logs": [
            {
                "timestamp": "2026-05-01T14:30:00Z",
                "source": "vpc-flow-logs",
                "text": (
                    "Traffic between order-service and postgres-primary: "
                    "normal volume, no drops"
                ),
            },
            {
                "timestamp": "2026-05-01T14:30:00Z",
                "source": "dns-resolver/query.log",
                "text": (
                    "postgres-primary.internal resolved to 10.0.2.15 "
                    "in 1ms — no errors"
                ),
            },
        ],
    },
    "expected_root_cause": (
        "An unoptimized analytics query (full sequential scan on 12M rows, "
        "no index usage) running without a statement_timeout consumed 40 of "
        "100 database connections for 47+ minutes. This exhausted the "
        "connection pool, causing the order-management service to queue "
        "requests until they timed out at 30 seconds."
    ),
    "expected_iterations": 3,
}


# ---------------------------------------------------------------------------
# Scenario 3 — TLS Certificate Expiry (Partial Rotation Failure)
# ---------------------------------------------------------------------------
# Root cause chain: automated cert rotation missed 2 of 5 backends →
# expired certs on backend-3 and backend-5 → TLS handshake failures →
# load balancer marks those backends unhealthy → intermittent 502 errors
# when requests hash to the bad backends.

CERTIFICATE_EXPIRY: dict = {
    "problem_statement": (
        "Intermittent HTTP 502 errors from the API gateway. "
        "Approximately 40% of requests fail, while 60% succeed. "
        "No recent deployments."
    ),
    "evidence_sources": {
        "app-logs": [
            {
                "timestamp": "2026-05-02T08:15:01Z",
                "source": "api-gateway.log",
                "text": (
                    "502 Bad Gateway — upstream TLS handshake failed "
                    "for backend-3:443"
                ),
            },
            {
                "timestamp": "2026-05-02T08:15:03Z",
                "source": "api-gateway.log",
                "text": (
                    "502 Bad Gateway — upstream TLS handshake failed "
                    "for backend-5:443"
                ),
            },
            {
                "timestamp": "2026-05-02T08:15:02Z",
                "source": "api-gateway.log",
                "text": "200 OK — request routed to backend-1:443 (82ms)",
            },
            {
                "timestamp": "2026-05-02T08:15:04Z",
                "source": "api-gateway.log",
                "text": "200 OK — request routed to backend-2:443 (91ms)",
            },
            {
                "timestamp": "2026-05-02T08:15:05Z",
                "source": "api-gateway.log",
                "text": "200 OK — request routed to backend-4:443 (76ms)",
            },
            {
                "timestamp": "2026-05-02T08:15:06Z",
                "source": "api-gateway.log",
                "text": (
                    "502 Bad Gateway — upstream TLS handshake failed "
                    "for backend-5:443"
                ),
            },
        ],
        "tls-certs": [
            {
                "timestamp": "2026-05-02T08:10:00Z",
                "source": "cert-manager/status",
                "text": (
                    "backend-1: cert serial=AF32..01, "
                    "expires=2026-11-01T00:00:00Z — VALID"
                ),
            },
            {
                "timestamp": "2026-05-02T08:10:00Z",
                "source": "cert-manager/status",
                "text": (
                    "backend-2: cert serial=AF32..02, "
                    "expires=2026-11-01T00:00:00Z — VALID"
                ),
            },
            {
                "timestamp": "2026-05-02T08:10:00Z",
                "source": "cert-manager/status",
                "text": (
                    "backend-3: cert serial=BF21..03, "
                    "expires=2026-05-01T23:59:59Z — EXPIRED"
                ),
            },
            {
                "timestamp": "2026-05-02T08:10:00Z",
                "source": "cert-manager/status",
                "text": (
                    "backend-4: cert serial=AF32..04, "
                    "expires=2026-11-01T00:00:00Z — VALID"
                ),
            },
            {
                "timestamp": "2026-05-02T08:10:00Z",
                "source": "cert-manager/status",
                "text": (
                    "backend-5: cert serial=BF21..05, "
                    "expires=2026-05-01T23:59:59Z — EXPIRED"
                ),
            },
            {
                "timestamp": "2026-05-02T08:10:00Z",
                "source": "cert-manager/rotation-log",
                "text": (
                    "Rotation job cert-rotate-20260501 completed: "
                    "3 of 5 backends updated successfully. "
                    "backend-3 FAILED (connection refused during deploy), "
                    "backend-5 FAILED (deploy agent not running)"
                ),
            },
        ],
        "lb-health": [
            {
                "timestamp": "2026-05-02T08:14:00Z",
                "source": "load-balancer/health",
                "text": (
                    "Backend pool health: backend-1 UP, backend-2 UP, "
                    "backend-3 DOWN (TLS error), backend-4 UP, "
                    "backend-5 DOWN (TLS error)"
                ),
            },
            {
                "timestamp": "2026-05-02T08:14:00Z",
                "source": "load-balancer/config",
                "text": (
                    "Routing algorithm: round-robin across 5 backends. "
                    "Health-check interval: 30s, threshold: 2 failures"
                ),
            },
            {
                "timestamp": "2026-05-02T08:00:00Z",
                "source": "load-balancer/health",
                "text": (
                    "Backend pool health: all 5 backends UP "
                    "(last full-healthy: 2026-05-01T23:58:00Z)"
                ),
            },
        ],
        "network-logs": [
            {
                "timestamp": "2026-05-02T08:15:00Z",
                "source": "firewall/syslog",
                "text": "No blocked connections to backend subnet 10.0.5.0/24",
            },
            {
                "timestamp": "2026-05-02T08:15:00Z",
                "source": "switch-core/syslog",
                "text": (
                    "All uplinks to backend rack operational — "
                    "no CRC errors, no drops"
                ),
            },
            {
                "timestamp": "2026-05-02T08:14:30Z",
                "source": "dns-resolver/query.log",
                "text": (
                    "api.example.com resolved correctly to load-balancer VIP "
                    "— TTL 300s, no NXDOMAIN"
                ),
            },
        ],
        "deploy-history": [
            {
                "timestamp": "2026-04-28T16:00:00Z",
                "source": "deploy-system/history",
                "text": (
                    "Last deployment: api-service v2.14.3 rolled out "
                    "4 days ago — all backends verified healthy post-deploy"
                ),
            },
            {
                "timestamp": "2026-05-01T02:00:00Z",
                "source": "deploy-system/cron",
                "text": (
                    "Scheduled cert rotation job cert-rotate-20260501 "
                    "triggered at 02:00 UTC"
                ),
            },
        ],
    },
    "expected_root_cause": (
        "The automated TLS certificate rotation job (cert-rotate-20260501) "
        "failed to update 2 of 5 backend servers (backend-3: connection "
        "refused, backend-5: deploy agent not running). The old certificates "
        "expired on 2026-05-01, causing TLS handshake failures for requests "
        "routed to those backends, producing intermittent 502 errors "
        "proportional to the 2/5 unhealthy ratio."
    ),
    "expected_iterations": 3,
}


# ---------------------------------------------------------------------------
# Convenience — all scenarios in one dict for programmatic iteration
# ---------------------------------------------------------------------------
ALL_SCENARIOS: dict[str, dict] = {
    "kubernetes_disk_pressure": KUBERNETES_DISK_PRESSURE,
    "database_connection_pool": DATABASE_CONNECTION_POOL,
    "certificate_expiry": CERTIFICATE_EXPIRY,
}
