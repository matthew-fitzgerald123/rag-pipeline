output "alb_dns" {
  value = aws_lb.main.dns_name
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}
