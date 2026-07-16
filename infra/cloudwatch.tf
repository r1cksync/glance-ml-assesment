# Log group for the API container (awslogs driver + EMF metrics) and the
# operations dashboard. EMF: the API writes structured metric logs into this
# group; CloudWatch materializes them as the FashionRetrieval namespace
# (LatencyMs / Requests, dimension: Route) at no extra ingestion cost.

resource "aws_cloudwatch_log_group" "api" {
  name              = local.log_group
  retention_in_days = 7
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = var.project

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "API latency (ms) — p50 / p95 by route"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          metrics = [
            [{ expression = "SEARCH('{FashionRetrieval,Route} MetricName=\"LatencyMs\"', 'p50', 300)", id = "p50", label = "p50" }],
            [{ expression = "SEARCH('{FashionRetrieval,Route} MetricName=\"LatencyMs\"', 'p95', 300)", id = "p95", label = "p95" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Requests (sum) by route"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          metrics = [
            [{ expression = "SEARCH('{FashionRetrieval,Route} MetricName=\"Requests\"', 'Sum', 300)", id = "req", label = "requests" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "EC2 CPU utilization"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          metrics = [
            ["AWS/EC2", "CPUUtilization", "InstanceId", aws_instance.api.id, { stat = "Average" }],
          ]
        }
      },
      {
        type   = "log"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "API logs (tail)"
          region = var.aws_region
          view   = "table"
          query  = "SOURCE '${aws_cloudwatch_log_group.api.name}' | fields @timestamp, @message | sort @timestamp desc | limit 100"
        }
      },
    ]
  })
}
