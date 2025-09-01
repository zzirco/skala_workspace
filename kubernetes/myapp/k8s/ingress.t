apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rewrite-target: /$2
  name: ${USER_NAME}-${SERVICE_NAME}
  namespace: ${NAMESPACE}
spec:
  ingressClassName: public-nginx
  rules:
  - host: ingress.skala25a.project.skala-ai.com
    http:
      paths:
      - backend:
          service:
            name: ${USER_NAME}-myfirst-api-server
            port:
              number: 8080
        path: /${USER_NAME}(/|$)(.*)
        pathType: ImplementationSpecific
  tls:
  - hosts:
    - 'ingress.skala25a.project.skala-ai.com'
    secretName: skala25-project-tls-cert
