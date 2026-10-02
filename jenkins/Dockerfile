# Jenkins controller with the tools the pipeline needs: python3/venv, docker CLI, kubectl, AWS CLI v2, git
FROM jenkins/jenkins:lts-jdk17
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip docker.io curl ca-certificates git unzip \
 && curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x /usr/local/bin/kubectl \
 && curl -fsSLo /tmp/awscli.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscli.zip \
 && rm -rf /var/lib/apt/lists/*
