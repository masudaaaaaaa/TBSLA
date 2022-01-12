#!/bin/sh
set -e

echo -I${INSTALL_DIR}/tbsla/include > ${INSTALL_DIR}/yml/230/release/var/yml/DefaultExecutionCatalog/generators/XMP/lib/include_tbsla.txt
echo -L${INSTALL_DIR}/tbsla/lib -ltbsla -ltbsla_mpi >> ${INSTALL_DIR}/yml/230/release/var/yml/DefaultExecutionCatalog/generators/XMP/lib/include_tbsla.txt

cp -r src/yml/DefaultExecutionCatalog ${INSTALL_DIR}/yml/230/release/var/yml/

mkdir -p ${HOME}/.omrpc_registry
cat > ${HOME}/.omrpc_registry/hosts.xml <<-EOT
<?xml version="1.0" ?>
<OmniRpcConfig>
<Host name="localhost" arch="i386" os="linux">
<Agent invoker="mpi" />
<JobScheduler type="rr" maxjob="100000" />
</Host>
</OmniRpcConfig>
EOT

sed "s%.*<entry name=\"hostfileDir\"        value=\".*\" />%<entry name=\"hostfileDir\"        value=\"${HOME}/.omrpc_registry\" />%" -i ${INSTALL_DIR}/yml/230/release/etc/yml/mpi.xcf
