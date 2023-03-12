import React from 'react';
import {RiMenu3Line, RiCloseLine} from 'react-icons/ri'
import {AiFillGithub, AiFillLinkedin} from 'react-icons/ai'
import logo from '../../assets/eagleeye.jpeg';
import './navbar.css';

const Navbar = () => {
  return (
    <div className='website__navbar'>
      <div className='website__navbar-links'>
        <div className='website__navbar-links_logo'>
          <img src={logo} alt='logo'></img>
        </div>
        <div className='website__navbar-links_container'>
          <p><a href='#home'>Home</a></p>
          <p><a href='#aboutme'>About Me</a></p>
          <p><a href='#interests'>Interests</a></p>
          <p><a href='#projects'>Projects</a></p>
        </div>
      </div>
      <div className='website__navbar-sign'>
        <a href='https://github.com/Malav-P'><AiFillGithub size={27}/> </a>
        <a href='https://linkedin.com/in/malavp00'><AiFillLinkedin size={27} /> </a>
      </div>
    </div>
  )
}

export default Navbar