variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "rag-pipeline"
}

variable "db_instance_class" {
  type    = string
  default = "db.t3.micro"
}

variable "db_name" {
  type    = string
  default = "rag_pipeline"
}

variable "db_username" {
  type    = string
  default = "rag_pipeline"
}

variable "db_password" {
  type      = string
  sensitive = true
}
