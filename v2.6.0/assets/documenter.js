// Generated by Documenter.jl
requirejs.config({
  paths: {
    'highlight-julia': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.15.10/languages/julia.min',
    'headroom': 'https://cdnjs.cloudflare.com/ajax/libs/headroom/0.10.3/headroom.min',
    'jqueryui': 'https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min',
    'katex-auto-render': 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.11.1/contrib/auto-render.min',
    'jquery': 'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.4.1/jquery.min',
    'headroom-jquery': 'https://cdnjs.cloudflare.com/ajax/libs/headroom/0.10.3/jQuery.headroom.min',
    'katex': 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.11.1/katex.min',
    'highlight': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.15.10/highlight.min',
    'highlight-julia-repl': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.15.10/languages/julia-repl.min',
  },
  shim: {
  "highlight-julia": {
    "deps": [
      "highlight"
    ]
  },
  "katex-auto-render": {
    "deps": [
      "katex"
    ]
  },
  "headroom-jquery": {
    "deps": [
      "jquery",
      "headroom"
    ]
  },
  "highlight-julia-repl": {
    "deps": [
      "highlight"
    ]
  }
}
});
////////////////////////////////////////////////////////////////////////////////
require(['jquery', 'katex', 'katex-auto-render'], function($, katex, renderMathInElement) {
$(document).ready(function() {
  renderMathInElement(
    document.body,
    {
  "delimiters": [
    {
      "left": "$",
      "right": "$",
      "display": false
    },
    {
      "left": "$$",
      "right": "$$",
      "display": true
    },
    {
      "left": "\\[",
      "right": "\\]",
      "display": true
    }
  ]
}

  );
})

})
////////////////////////////////////////////////////////////////////////////////
require(['jquery', 'highlight', 'highlight-julia', 'highlight-julia-repl'], function($, hljs) {
$(document).ready(function() {
    hljs.initHighlighting();
})

})
////////////////////////////////////////////////////////////////////////////////
require(['jquery', 'headroom', 'headroom-jquery'], function($, Headroom) {

// Manages the top navigation bar (hides it when the user starts scrolling down on the
// mobile).
window.Headroom = Headroom; // work around buggy module loading?
$(document).ready(function() {
  $('#documenter .docs-navbar').headroom({
    "tolerance": {"up": 10, "down": 10},
  });
})

})
////////////////////////////////////////////////////////////////////////////////
require(['jquery'], function($) {

// Modal settings dialog
$(document).ready(function() {
  var settings = $('#documenter-settings');
  $('#documenter-settings-button').click(function(){
    settings.toggleClass('is-active');
  });
  // Close the dialog if X is clicked
  $('#documenter-settings button.delete').click(function(){
    settings.removeClass('is-active');
  });
  // Close dialog if ESC is pressed
  $(document).keyup(function(e) {
    if (e.keyCode == 27) settings.removeClass('is-active');
  });
});

})
////////////////////////////////////////////////////////////////////////////////
require(['jquery'], function($) {

// Manages the showing and hiding of the sidebar.
$(document).ready(function() {
  var sidebar = $("#documenter > .docs-sidebar");
  var sidebar_button = $("#documenter-sidebar-button")
  sidebar_button.click(function(ev) {
    ev.preventDefault();
    sidebar.toggleClass('visible');
    if (sidebar.hasClass('visible')) {
      // Makes sure that the current menu item is visible in the sidebar.
      $("#documenter .docs-menu a.is-active").focus();
    }
  });
  $("#documenter > .docs-main").bind('click', function(ev) {
    if ($(ev.target).is(sidebar_button)) {
      return;
    }
    if (sidebar.hasClass('visible')) {
      sidebar.removeClass('visible');
    }
  });
})

// Resizes the package name / sitename in the sidebar if it is too wide.
// Inspired by: https://github.com/davatron5000/FitText.js
$(document).ready(function() {
  e = $("#documenter .docs-autofit");
  function resize() {
    var L = parseInt(e.css('max-width'), 10);
    var L0 = e.width();
    if(L0 > L) {
      var h0 = parseInt(e.css('font-size'), 10);
      e.css('font-size', L * h0 / L0);
      // TODO: make sure it survives resizes?
    }
  }
  // call once and then register events
  resize();
  $(window).resize(resize);
  $(window).on('orientationchange', resize);
});

// Scroll the navigation bar to the currently selected menu item
$(document).ready(function() {
  var sidebar = $("#documenter .docs-menu").get(0);
  var active = $("#documenter .docs-menu .is-active").get(0);
  if(typeof active !== 'undefined') {
    sidebar.scrollTop = active.offsetTop - sidebar.offsetTop - 15;
  }
})

})
////////////////////////////////////////////////////////////////////////////////
require(['jquery'], function($) {

function set_theme(theme) {
  var active = null;
  var disabled = [];
  for (var i = 0; i < document.styleSheets.length; i++) {
    var ss = document.styleSheets[i];
    var themename = ss.ownerNode.getAttribute("data-theme-name");
    if(themename === null) continue; // ignore non-theme stylesheets
    // Find the active theme
    if(themename === theme) active = ss;
    else disabled.push(ss);
  }
  if(active !== null) {
    active.disabled = false;
    if(active.ownerNode.getAttribute("data-theme-primary") === null) {
      document.getElementsByTagName('html')[0].className = "theme--" + theme;
    } else {
      document.getElementsByTagName('html')[0].className = "";
    }
    disabled.forEach(function(ss){
      ss.disabled = true;
    });
  }

  // Store the theme in localStorage
  if(typeof(window.localStorage) !== "undefined") {
    window.localStorage.setItem("documenter-theme", theme);
  } else {
    console.error("Browser does not support window.localStorage");
  }
}

// Theme picker setup
$(document).ready(function() {
  // onchange callback
  $('#documenter-themepicker').change(function themepick_callback(ev){
    var themename = $('#documenter-themepicker option:selected').attr('value');
    set_theme(themename);
  });

  // Make sure that the themepicker displays the correct theme when the theme is retrieved
  // from localStorage
  if(typeof(window.localStorage) !== "undefined") {
    var theme =  window.localStorage.getItem("documenter-theme");
    if(theme !== null) {
      $('#documenter-themepicker option').each(function(i,e) {
        e.selected = (e.value === theme);
      })
    }
  }
})

})
////////////////////////////////////////////////////////////////////////////////
require(['jquery'], function($) {

// update the version selector with info from the siteinfo.js and ../versions.js files
$(document).ready(function() {
  var version_selector = $("#documenter .docs-version-selector");
  var version_selector_select = $("#documenter .docs-version-selector select");

  version_selector_select.change(function(x) {
    target_href = version_selector_select.children("option:selected").get(0).value;
    window.location.href = target_href;
  });

  // add the current version to the selector based on siteinfo.js, but only if the selector is empty
  if (typeof DOCUMENTER_CURRENT_VERSION !== 'undefined' && $('#version-selector > option').length == 0) {
    var option = $("<option value='#' selected='selected'>" + DOCUMENTER_CURRENT_VERSION + "</option>");
    version_selector_select.append(option);
  }

  if (typeof DOC_VERSIONS !== 'undefined') {
    var existing_versions = version_selector_select.children("option");
    var existing_versions_texts = existing_versions.map(function(i,x){return x.text});
    DOC_VERSIONS.forEach(function(each) {
      var version_url = documenterBaseURL + "/../" + each;
      var existing_id = $.inArray(each, existing_versions_texts);
      // if not already in the version selector, add it as a new option,
      // otherwise update the old option with the URL and enable it
      if (existing_id == -1) {
        var option = $("<option value='" + version_url + "'>" + each + "</option>");
        version_selector_select.append(option);
      } else {
        var option = existing_versions[existing_id];
        option.value = version_url;
        option.disabled = false;
      }
    });
  }

  // only show the version selector if the selector has been populated
  if (version_selector_select.children("option").length > 0) {
    version_selector.toggleClass("visible");
  }
})

})
