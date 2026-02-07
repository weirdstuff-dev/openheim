use super::types::{AgentConfig, AppConfig};
use crate::error::{Error, Result};

impl AppConfig {
    pub fn resolve(&self, model_name: Option<&str>) -> Result<AgentConfig> {
        match model_name {
            Some(name) => self.resolve_model(name),
            None => self.resolve_default(),
        }
    }
    
    fn resolve_default(&self) -> Result<AgentConfig> {
        let provider = self.providers.get(&self.default_provider).ok_or_else(|| {
            Error::config(format!(
                "Default provider '{}' not found in config. Available providers: {}",
                self.default_provider,
                self.provider_names()
            ))
        })?;
        Ok(AgentConfig {
            provider_name: self.default_provider.clone(),
            api_base: provider.api_base.clone(),
            api_key: provider.resolve_api_key(),
            model: provider.default_model.clone(),
            max_iterations: self.max_iterations,
        })
    }

    fn resolve_model(&self, model_name: &str) -> Result<AgentConfig> {
        for (name, provider) in &self.providers {
            if provider.models.contains(&model_name.to_string()) {
                return Ok(AgentConfig {
                    provider_name: name.clone(),
                    api_base: provider.api_base.clone(),
                    api_key: provider.resolve_api_key(),
                    model: model_name.to_string(),
                    max_iterations: self.max_iterations,
                });
            }
        }
        Err(Error::config(format!(
            "Model '{}' not found in any provider. Run `openheim --list` to see available models.",
            model_name
        )))
    }

    pub fn list_models(&self) -> Result<String> {
        if self.providers.is_empty() {
            return Err(Error::config(
                "No providers configured. Edit your config file to add at least one provider.",
            ));
        }

        let mut out = String::from("Configured providers:\n");
        for (name, provider) in &self.providers {
            let is_default = name == &self.default_provider;
            let suffix = if is_default { " (default)" } else { "" };
            out.push_str(&format!("\n  {}{}\n", name, suffix));
            out.push_str(&format!("    api_base: {}\n", provider.api_base));

            let models: Vec<String> = provider
                .models
                .iter()
                .map(|m| {
                    if m == &provider.default_model {
                        format!("{} (default)", m)
                    } else {
                        m.clone()
                    }
                })
                .collect();
            out.push_str(&format!("    models:   {}\n", models.join(", ")));
        }
        Ok(out)
    }

    fn provider_names(&self) -> String {
        self.providers
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    }
}
