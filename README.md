# UNSW-THESIS-HUB
Investigating Quantization-Induced Vulnerabilities in Deep Neural Networks: Backdoor Data Poisoning and Detection Strategies 


Deep Learning Models and their deployment in resource-constrained environments are
increasingly important to extend their utility into edge devices, mobile platforms, and
other low-resource systems. Quantization methods, which decrease the precision of
model weights and activations, are widely employed to optimize model size and run-
time efficiency. These low-precision models introduce underexplored vulnerabilities,
especially in contexts like data poisoning. Some of these attacks can leverage quantiza-
tion to deteriorate model performance, introduce computational inefficiencies, or elicit
malicious behaviors in ways that may not show in full-precision versions.
This thesis investigates the security implications of quantized DNN, emphasizing back-
door attacks. We investigate how a robust DNN at full precision may be susceptible to
backdoor attacks following quantization.
For this reason, we will implement customized DNN architectures mainly focusing on
Large Language Models and carefully investigate their behavior both before and after
quantization, practically exploring several quantization strategies that aim to provide
insights about the vulnerability patterns. We also design and evaluate detection strate-
gies to identify backdoor attacks in quantized models, leveraging theoretical analysis
and empirical validation.
This work investigates how quantization methods interact with data poisonings, with
a more critical concern for the vulnerabilities of quantized LLMs and proposing robust
methods of detection that can help in developing safer and more reliable quantiza-
tion techniques for deploying LLMs and other types of DNNs via resource-constrained,
safety-critical applications like health care, autonomous systems, and financial services.
