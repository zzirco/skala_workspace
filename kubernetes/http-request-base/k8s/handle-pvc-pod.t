apiVersion: v1
kind: Pod
metadata:
  name: ${USER_NAME}-pvc-handler-pod
  namespace: ${NAMESPACE}
spec:
  containers:
  - name: ubuntu
    image: ubuntu:22.04
    command: ["/bin/bash","-lc","apt-get update && apt-get install -y vim sudo && tail -f /dev/null"]
    volumeMounts:
    - name: efs-config
      mountPath: /config
  volumes:
  - name: efs-config
    persistentVolumeClaim:
      claimName: ${USER_NAME}-efs-sc-${SERVICE_NAME}-pvc
